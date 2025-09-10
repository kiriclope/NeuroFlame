import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True

from src.configuration import Configuration, init_time_const
from src.connectivity import Connectivity
from src.activation import Activation
from src.plasticity import Plasticity
from src.lr_utils import LowRankWeights, clamp_tensor, normalize_tensor

from src.ff_input import live_ff_input, init_ff_input, rl_ff_udpdate
from src.utils import set_seed, clear_cache, print_activity

import warnings

warnings.filterwarnings("ignore")


class Network(nn.Module):
    """
    Class: Network
    Creates a recurrent network of rate units with customizable connectivity and dynamics.
    The network can be trained with standard torch optimization technics.
    Parameters:
        conf_name: str, name of a .yml file contaning the model's parameters.
        repo_root: str, root path of the NeuroFlame repository.
        **kwargs: **dict, any parameter in conf_file can be passed here
                             and will then be overwritten.
    Returns:
           rates: tensorfloat of size (N_BATCH, N_STEPS or 1, N_NEURON).
    """

    def __init__(self, conf_name, repo_root, **kwargs):
        super().__init__()

        # Load parameters from configuration file and create networks constants
        config = Configuration(conf_name, repo_root)(**kwargs)
        self.__dict__.update(config.__dict__)

        # Initialize weight matrix
        self.initWeights()
        # reset seed
        set_seed(-1)

        # Initialize low rank connectivity for training
        if self.LR_TRAIN:

            self.low_rank = LowRankWeights(
                self.Na[0], # N_NEURON
                self.Na,
                self.slices,
                self.RANK,
                self.LR_MN,
                self.LR_READOUT,
                self.LR_INI,
                self.LR_UeqV,
                self.device,
            )

        # Add STP
        if self.IF_STP:
            self.initSTP()

        clear_cache()

    def initWeights(self):
        """
        Initializes the connectivity matrix self.Wab.
        Loops over blocks to create the full matrix.
        Relies on class Connectivity from connectivity.py
        """

        # in pytorch, Wij is i to j.
        self.Wab_train = 0
        if self.ODR_TRAIN:
            if self.TRAIN_EI==0:
                self.Wab_train = nn.Parameter(torch.randn((self.Na[0], self.Na[0]),
                                                          device=self.device)* 0.001)
            else:
                self.Wab_train = nn.Parameter(torch.randn((self.N_NEURON, self.N_NEURON),
                                                          device=self.device)* 0.001)

                self.train_mask = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)

                for i_pop in range(self.N_POP):
                    for j_pop in range(self.N_POP):
                        self.train_mask[self.slices[i_pop], self.slices[j_pop]] = self.IS_TRAIN[i_pop][j_pop]

        self.Wab_T = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)

        # Creates connetivity matrix in blocks
        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):
                weight_mat = Connectivity(
                    self.Na[i_pop], self.Na[j_pop], self.Ka[j_pop], device=self.device
                )

                weights = weight_mat(
                    self.CON_TYPE,
                    self.PROBA_TYPE[i_pop][j_pop],
                    kappa=self.KAPPA[i_pop][j_pop],
                    phase=self.PHASE,
                    sigma=self.SIGMA[i_pop][j_pop],
                    lr_mean=self.LR_MEAN,
                    lr_cov=self.LR_COV,
                    ksi=self.PHI0,
                )

                if i_pop==0 and j_pop==0:
                    self.sparse_mask = weights

                self.Wab_T.data[self.slices[i_pop], self.slices[j_pop]] = (
                    self.Jab[i_pop][j_pop] * weights
                )

        if self.SPARSE == "full":
            self.Wab_T = self.Wab_T.T.to_sparse()
        elif self.SPARSE == "semi":
            self.Wab_T = to_sparse_semi_structured(self.Wab_T)
        else:
            self.Wab_T = self.Wab_T.T

        # self.register_buffer('Wab_T', self.Wab_T)

    def initSTP(self):
        """Creates stp model for population 0"""

        self.J_STP = torch.tensor(self.J_STP, device=self.device)
        if self.training:
            self.J_STP = nn.Parameter(self.J_STP)

        # NEED .clone() here otherwise BAD THINGS HAPPEN !!!
        self.W_stp_T = [self.GAIN * self.Wab_T[self.slices[0], self.slices[0]].clone()
                        / self.Jab[0, 0]
                        / torch.sqrt(self.Ka[0])]

        if self.TEST_I_STP:
            self.W_stp_T.append(self.Wab_T[self.slices[0], self.slices[1]].clone() / torch.abs(self.Jab[1, 0]))

        if self.LR_TYPE == 'full':
            self.W_stp_T[0] = 1.0 / self.Na[0]

        # remove non plastic connections
        self.Wab_T.data[self.slices[0], self.slices[0]] = 0.0

        if self.TEST_I_STP:
            # self.Wab_T.data[self.slices[1], self.slices[1]] = 0.0
            self.Wab_T.data[self.slices[0], self.slices[1]] = 0.0

        if self.TRAIN_EI:
            self.train_mask[self.slices[0], self.slices[0]] = 0.0


    def init_ff_input(self):
        return init_ff_input(self)


    def initRates(self, ff_input=None):
        if ff_input is None:
            if self.VERBOSE:
                print("Generating ff input")

            ff_input = init_ff_input(self)
        else:
            ff_input.to(self.device)

        self.N_BATCH = ff_input.shape[0]
        thresh = self.thresh[:ff_input.shape[0]]

        rec_input = torch.randn((self.IF_NMDA + 1, self.N_BATCH, self.N_NEURON), device=self.device)

        if self.LIVE_FF_UPDATE:
            ff = ff_input
        else:
            ff = ff_input[:, 0]

        if self.IF_FF_ADAPT:
            self.thresh_ff = torch.zeros_like(ff_input[:, 0])

        rates = Activation()(ff + rec_input[0], func_name=self.TF_TYPE, thresh=thresh)

        return rates, ff_input, rec_input, thresh


    def update_dynamics(self, rates, ff_input, rec_input, Wab_T, W_stp_T, thresh):
        """Updates the dynamics of the model at each timestep"""

        # update hidden state
        # if self.SPARSE == "full":
        #     hidden = torch.sparse.mm(rates, Wab_T)
        # elif self.SPARSE == "semi":
        #     hidden = (Wab_T @ rates.T).T
        # else:
        hidden = rates @ Wab_T # here @ is sum_j rj * Wji hence it's j to i (row are presyn)

        # update stp variables
        if self.IF_STP:
            Aux = self.stp[0](rates[:, self.slices[0]])  # Aux is now u * x * rates
            hidden_stp = Aux @ W_stp_T[0]
            hidden[:, self.slices[0]] = hidden[:, self.slices[0]] + hidden_stp

            if self.TEST_I_STP:
                Aux = self.stp[1](rates[:, self.slices[0]])  # Aux is now u * x * rates
                hidden_stp = Aux @ W_stp_T[1]
                hidden[:, self.slices[1]] = hidden[:, self.slices[1]] + hidden_stp

        if self.IF_FF_STP:
            hidden_ff_stp = torch.sign(ff_input[:, self.slices[0]]) * self.ff_stp(nn.ReLU()(ff_input[:, self.slices[0]]))
            ff_input[:, self.slices[0]] = ff_input[:, self.slices[0]] + hidden_ff_stp

        # update batched EtoE
        if self.IF_BATCH_J:
            hidden[:, self.slices[0]].add_(self.Jab_batch * rates[:, self.slices[0]] @ self.W_batch_T)

        # update reccurent input
        if self.SYN_DYN:
            # exponential euler method
            rec_input[0] = rec_input[0] * self.EXP_DT_TAU_SYN + hidden * (1.0 - self.EXP_DT_TAU_SYN)
        else:
            rec_input[0] = hidden

        if self.IF_FF_ADAPT:
            # ff_input = ff_input / (1.0 + self.thresh_ff)
            ff_input = torch.sign(ff_input) * nn.ReLU()(ff_input - self.thresh_ff)
            self.thresh_ff = self.thresh_ff * self.EXP_FF_ADAPT + nn.ReLU()(ff_input) * self.A_FF_ADAPT * (1.0-self.EXP_FF_ADAPT)

        # compute net input
        net_input = ff_input + rec_input[0]

        if self.IF_NMDA:
            hidden = rates[:, self.slices[0]] @ Wab_T[self.slices[0]]

            if self.IF_STP:
                hidden[:, self.slices[0]] = hidden[:, self.slices[0]] + hidden_stp

            # exponential euler method
            rec_input[1] = rec_input[1] * self.EXP_DT_TAU_NMDA + self.R_NMDA * hidden * (1.0 - self.EXP_DT_TAU_NMDA)

            net_input = net_input + rec_input[1]

        # compute non linearity
        non_linear = Activation()(net_input, func_name=self.TF_TYPE, thresh=thresh)

        # update rates
        if self.RATE_DYN:
            # exponential euler method
            rates = rates * self.EXP_DT_TAU + non_linear * (1.0 - self.EXP_DT_TAU)
        else:
            rates = non_linear

        # clear_cache()

        # adaptation
        if self.IF_ADAPT:
            thresh = thresh * self.EXP_ADAPT + rates.detach() * self.A_ADAPT * (1.0 - self.EXP_ADAPT)

        return rates, rec_input, thresh


    def forward(self, ff_input=None, REC_LAST_ONLY=0, RET_FF=0, RET_STP=0, RET_REC=0, IF_INIT=1):
        """
        Main method of Network class, runs networks dynamics over set of timesteps
        and returns rates at each time point or just the last time point.
        args:
        :param ff_input: float (N_BATCH, N_STEP, N_NEURONS), ff inputs into the network.
        :param REC_LAST_ONLY: bool, wether to record the last timestep only.
        rates_list:
        :param rates_list: float (N_BATCH, N_STEP or 1, N_NEURONS), rates of the neurons.
        """

        # Initialization (if  ff_input is None, ff_input is generated)
        if IF_INIT:
            rates, ff_input, rec_input, thresh = self.initRates(ff_input)
        else:
            rates, rec_input, thresh = self.rates_last, self.rec_input_last, self.thresh_last

        # NEED .clone() here otherwise BAD THINGS HAPPEN
        if self.IF_BATCH_J:
            self.W_batch_T = self.Wab_T[self.slices[0], self.slices[0]].clone() / self.Jab[0, 0]
            self.Wab_T.data[self.slices[0], self.slices[0]] = 0

        # Add STP
        if self.IF_STP:
            self.stp = []
            # Need this here otherwise autograd complains
            self.stp.append(Plasticity(self.USE[0], self.TAU_FAC[0], self.TAU_REC[0], self.DT,
                                       (self.N_BATCH, self.Na[0]),
                                       STP_TYPE=self.STP_TYPE,
                                       IF_INIT=IF_INIT,
                                       device=self.device,
                                       ))

            if self.TEST_I_STP:
                self.stp.append(Plasticity(self.USE[1], self.TAU_FAC[1], self.TAU_REC[1], self.DT,
                                           (self.N_BATCH, self.Na[0]),
                                           STP_TYPE=self.STP_TYPE,
                                           IF_INIT=IF_INIT,
                                           device=self.device,
                                           ))

            # previous state is loaded
            if IF_INIT == 0:
                self.stp[0].u_stp = self.u_stp_last
                self.stp[0].x_stp = self.x_stp_last

            self.x_list, self.u_list = [], []

        if self.IF_FF_STP:
            self.ff_stp = Plasticity(self.FF_USE,self.TAU_FF_FAC, self.TAU_FF_REC, self.DT,
                                     (self.N_BATCH, self.Na[0]),
                                     STP_TYPE=self.STP_TYPE,
                                     IF_INIT=IF_INIT,
                                     device=self.device,
                                     )

        Wab_T = self.Wab_T

        if self.LR_TRAIN:
            self.Wab_train = self.low_rank(self.LR_NORM, self.LR_CLAMP)

        if self.IF_STP:
            W_stp_T = [self.GAIN * self.J_STP * (self.W_stp_T[0] + self.Wab_train / self.Na[0])]

            if self.CLAMP:
                W_stp_T[0] = clamp_tensor(W_stp_T[0], 0, self.slices)

            if self.TEST_I_STP:
                W_stp_T.append(self.GAIN * self.J_STP * self.W_stp_T[1])

        if self.IF_OPTO:
            # rand_idx = torch.randperm(W_stp_T.size(0))[:self.N_OPTO]
            # W_stp_T[:, rand_idx] = 0

            rand_idx = torch.randperm(W_stp_T[0].size(0))[:self.N_OPTO]
            W_stp_T[0][rand_idx] = 0

            # _, idx = torch.sort(self.low_rank.V[:,1])
            # W_stp_T[idx[:self.N_OPTO]] = 0

        # Moving average
        mv_rates, mv_ff, mv_rec = 0, 0, 0
        rates_list, ff_list, rec_list = [], [], []

        # Temporal loop
        for step in range(self.N_STEPS):
            # add noise to the rates
            if self.RATE_NOISE:
                rate_noise = torch.randn((self.N_BATCH, self.N_NEURON), device=self.device)
                rates = rates + rate_noise * self.VAR_RATE

            # create ff input at each time step
            if self.LIVE_FF_UPDATE:
                ff_input = live_ff_input(self, step, ff_input)
                rates, rec_input, thresh = self.update_dynamics(rates, ff_input, rec_input, Wab_T, W_stp_T, thresh)
            else:
                rates, rec_input, thresh = self.update_dynamics(rates, ff_input[:, step], rec_input, Wab_T, W_stp_T, thresh)

            # update moving average
            mv_rates += rates
            if self.LIVE_FF_UPDATE and RET_FF:
                mv_ff += ff_input

            if RET_REC:
                mv_rec += rec_input

            # Reset moving average to start at 0
            if step == self.N_STEADY - self.N_WINDOW - 1:
                mv_rates = 0.0
                mv_ff = 0.0
                mv_rec = 0.0

            # update output every N_WINDOW steps
            if step >= self.N_STEADY:
                if step % self.N_WINDOW == 0:
                    if self.VERBOSE:
                        print_activity(self, step, rates)

                    if not REC_LAST_ONLY:
                        # rates_list.append(mv_rates / self.N_WINDOW)
                        rates_list.append(mv_rates[..., self.slices[0]] / self.N_WINDOW)

                        if self.LIVE_FF_UPDATE and RET_FF:
                            ff_list.append(mv_ff[..., self.slices[0]] / self.N_WINDOW)
                        if RET_REC:
                            rec_list.append(mv_rec[..., self.slices[0]] / self.N_WINDOW)
                        if self.IF_STP and RET_STP:
                            self.x_list.append(self.stp[0].x_stp)
                            self.u_list.append(self.stp[0].u_stp)

                    # Reset moving average
                    mv_rates = 0
                    mv_ff = 0
                    mv_rec = 0

        # makes serialisation easier treating it as epochs
        # we save the network state and run 2 trials at a time
        self.rates_last = rates
        self.rec_input_last = rec_input
        self.thresh_last = thresh

        if self.IF_STP:
            self.u_stp_last = self.stp[0].u_stp
            self.x_stp_last = self.stp[0].x_stp

        # returns last step
        # rates = rates[..., self.slices[0]]

        # returns full sequence
        if REC_LAST_ONLY == 0:
            # Stack list on 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
            rates = torch.stack(rates_list, dim=1)

            if RET_REC:
                self.rec_input = torch.stack(rec_list, dim=2)

            if self.LIVE_FF_UPDATE and RET_FF:  # returns ff input
                self.ff_input = torch.stack(ff_list, dim=1)

            if self.IF_STP and RET_STP:  # returns stp u and x
                self.u_list = torch.stack(self.u_list, dim=1)
                self.x_list = torch.stack(self.x_list, dim=1)

        if self.LR_TRAIN:
            self.readout = rates @ self.low_rank.V[self.slices[0]] / self.Na[0]
            if self.LR_READOUT==1:
                linear = self.low_rank.linear(self.dropout(rates)) / self.Na[0]
                self.readout = torch.cat((self.readout, linear), dim=-1)

        if self.LIVE_FF_UPDATE == 0 and RET_FF:
            self.ff_input = ff_input[..., self.slices[0]]

        clear_cache()

        return rates
