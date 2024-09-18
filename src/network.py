import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True


from src.configuration import Configuration
from src.connectivity import Connectivity
from src.activation import Activation
from src.plasticity import Plasticity
from src.lr_utils import LowRankWeights, clamp_tensor

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

        # Initialize low rank connectivity for training
        if self.LR_TRAIN:
            self.odors = torch.randn(
                (10, self.Na[0]),
                device=self.device,
            )

            self.low_rank = LowRankWeights(
                self.N_NEURON,
                self.Na,
                self.slices,
                self.RANK,
                self.LR_MN,
                self.LR_KAPPA,
                self.LR_BIAS,
                self.LR_READOUT,
                self.LR_FIX_READ,
                self.DROP_RATE,
                self.LR_MASK,
                self.LR_CLASS,
                self.device,
            )

        # Add STP
        if self.IF_STP:
            self.initSTP()

        # Reset the seed
        set_seed(0)
        clear_cache()

    def initWeights(self):
        """
        Initializes the connectivity matrix self.Wab.
        Scales weights Jab and loops over blocks to create the full matrix.
        Relies on class Connectivity from connetivity.py
        """

        # # Scale synaptic weights as 1/sqrt(K) for sparse nets
        # self.scaleWeights()

        # in pytorch, Wij is i to j.
        if self.ODR_TRAIN:
            self.Wab_train = nn.Parameter(torch.randn((self.Na[0], self.Na[0]),
                                                     device=self.device)* 0.01)

            self.odr_mask = torch.ones((self.N_NEURON, self.N_NEURON), device=self.device)
            self.odr_mask[self.slices[0], self.slices[0]] = 0.0

        self.register_buffer('Wab_T', torch.zeros((self.N_NEURON, self.N_NEURON),
                                                  device=self.device))

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

                self.Wab_T.data[self.slices[i_pop], self.slices[j_pop]] = (
                    self.Jab[i_pop][j_pop] * weights
                )

        # if self.ODR_TRAIN:
        #     weights = weight_mat(
        #         self.CON_TYPE,
        #         'cosine',
        #         kappa=torch.tensor(1.0).to(self.device),
        #         phase=self.PHASE,
        #         sigma=self.SIGMA[0][0],
        #         lr_mean=self.LR_MEAN,
        #         lr_cov=self.LR_COV,
        #         ksi=self.PHI0,
        #     )

        #     self.Wab_train.data = (
        #         self.Jab[0][0] * weights
        #     )

        del weights, weight_mat

        if self.SPARSE == "full":
            self.Wab_T = self.Wab_T.T.to_sparse()
        elif self.SPARSE == "semi":
            self.Wab_T = to_sparse_semi_structured(self.Wab_T)
        else:
            # take weights transpose for optim
            if self.ODR_TRAIN==0:
                self.Wab_T = self.Wab_T.T

        if self.LR_TRAIN==0:
            self.Wab_T = self.Wab_T

    def initSTP(self):
        """Creates stp model for population 0"""
        self.J_STP = torch.tensor(self.J_STP, device=self.device) * (
            self.GAIN / torch.sqrt(self.Ka[0])
        )

        # NEED .clone() here otherwise BAD THINGS HAPPEN !!!
        self.register_buffer('W_stp_T', torch.zeros((self.Na[0], self.Na[0]), device=self.device))

        self.W_stp_T = (
            self.Wab_T[self.slices[0], self.slices[0]].clone() / self.Jab[0, 0]
        )

        self.Wab_T.data[self.slices[0], self.slices[0]] = 0

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

        rec_input = torch.randn(
            (self.IF_NMDA + 1, self.N_BATCH, self.N_NEURON),
            device=self.device,
        )

        if self.LIVE_FF_UPDATE:
            rates = Activation()(
                ff_input + rec_input[0], func_name=self.TF_TYPE, thresh=self.thresh
            )
        else:
            rates = Activation()(
                ff_input[:, 0] + rec_input[0],
                func_name=self.TF_TYPE,
                thresh=self.thresh,
            )

        return rates, ff_input, rec_input

    def scaleWeights(self):
        # scaling recurrent weights Jab as 1 / sqrt(Kb)
        if self.VERBOSE:
            print("Jab", self.Jab)

        for i_pop in range(self.N_POP):
            self.Jab[:, i_pop] = self.Jab[:, i_pop] / torch.sqrt(self.Ka[i_pop])

        # scaling FF weights as sqrt(K0)
        if self.VERBOSE:
            print("Ja0", self.Ja0)

        # now inputs are scaled in init_ff_input unless live update
        if self.LIVE_FF_UPDATE:
            self.Ja0 = self.M0 * torch.sqrt(self.Ka[0]) * self.Ja0

        # scaling ff variance as 1 / sqrt(K0)
        self.VAR_FF.mul_(self.M0 / torch.sqrt(self.Ka[0]))

    def update_dynamics(self, rates, ff_input, rec_input, Wab_T, W_stp_T):
        """Updates the dynamics of the model at each timestep"""

        # update hidden state
        if self.SPARSE == "full":
            hidden = torch.sparse.mm(rates, Wab_T)
        elif self.SPARSE == "semi":
            hidden = (Wab_T @ rates.T).T
        else:
            hidden = rates @ Wab_T

        # update stp variables
        if self.IF_STP:
            Aux = self.stp(rates[:, self.slices[0]])  # Aux is now u * x * rates
            hidden_stp = self.J_STP * Aux @ W_stp_T  # / torch.sqrt(self.Ka[0])
            hidden[:, self.slices[0]] = hidden[:, self.slices[0]] + hidden_stp

        # update batched EtoE
        if self.IF_BATCH_J:
            hidden[:, self.slices[0]].add_(
                self.Jab_batch * rates[:, self.slices[0]] @ self.W_batch_T
            )

        # update reccurent input
        if self.SYN_DYN:
            rec_input[0] = rec_input[0] * self.EXP_DT_TAU_SYN + hidden * self.DT_TAU_SYN
        else:
            rec_input[0] = hidden

        # compute net input
        net_input = ff_input + rec_input[0]

        if self.IF_NMDA:
            hidden = rates[:, self.slices[0]] @ Wab_T[self.slices[0]]

            if self.IF_STP:
                hidden[:, self.slices[0]] = hidden[:, self.slices[0]] + hidden_stp

            rec_input[1] = (
                rec_input[1] * self.EXP_DT_TAU_NMDA
                + self.R_NMDA * hidden * self.DT_TAU_NMDA
            )

            net_input = net_input + rec_input[1]

        # compute non linearity
        non_linear = Activation()(net_input, func_name=self.TF_TYPE, thresh=0)

        # update rates
        if self.RATE_DYN:
            rates = rates * self.EXP_DT_TAU + non_linear * self.DT_TAU
        else:
            rates = non_linear

        # this makes autograd complain about the graph
        # if self.IF_ADAPT:
        #     self.thresh = self.thresh * self.EXP_DT_TAU_ADAPT
        #     self.thresh = self.thresh + nn.ReLU()(rates * self.A_ADAPT) * self.DT_TAU_ADAPT

        return rates, rec_input

    def forward(self, ff_input=None, REC_LAST_ONLY=0, RET_FF=0, RET_STP=0):
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
        rates, ff_input, rec_input = self.initRates(ff_input)

        # NEED .clone() here otherwise BAD THINGS HAPPEN
        if self.IF_BATCH_J:
            self.W_batch_T = (
                self.Wab_T[self.slices[0], self.slices[0]].clone() / self.Jab[0, 0]
            )

            self.Wab_T.data[self.slices[0], self.slices[0]] = 0

        # Add STP
        W_stp_T = None
        if self.IF_STP:
            # Need this here otherwise autograd complains
            self.stp = Plasticity(
                self.USE,
                self.TAU_FAC,
                self.TAU_REC,
                self.DT,
                (self.N_BATCH, self.Na[0]),
                STP_TYPE=self.STP_TYPE,
                device=self.device,
            )

            # self.Wab_T.data[self.slices[0], self.slices[0]] = 0
            self.x_list, self.u_list = [], []
            W_stp_T = self.W_stp_T

        if self.LR_TRAIN:
            self.lr = self.low_rank(self.LR_NORM, self.LR_CLAMP) * torch.sqrt(
                self.Ka[0]
            )

            # self.lr = self.low_rank(self.LR_NORM, self.LR_CLAMP)

            # this breaks autograd :s
            # self.odors[2] = self.low_rank.linear.weight[0]

            if self.IF_STP:
                W_stp_T = self.W_stp_T + self.lr[self.slices[0], self.slices[0]].T
                # W_stp_T = self.W_stp_T * (1.0 + self.lr[self.slices[0], self.slices[0]].T)
                W_stp_T = clamp_tensor(W_stp_T, 0, self.slices)

                Wab_T = self.Wab_T
            else:
                Wab_T = self.Wab_T + self.lr.T
                # Wab_T = self.Wab_T + self.Wab_T[0, 0] * self.lr.T
                # Wab_T = self.Wab_T * (1.0 + self.lr.T)

            # Wab_T = clamp_tensor(Wab_T, 0, self.slices)
            # Wab_T = clamp_tensor(Wab_T, 1, self.slices)
        elif self.ODR_TRAIN:
            if self.IF_STP:
                Wab_T = self.odr_mask * self.Wab_T
                W_stp_T = self.Wab_train / torch.sqrt(self.Na[0])
                # W_stp_T = clamp_tensor(W_stp_T, 0, self.slices)
            else:
                Wab_T = self.Wab_T
        else:
            Wab_T = self.Wab_T
            if self.IF_STP:
                W_stp_T = self.W_stp_T

        # Moving average
        mv_rates, mv_ff = 0, 0
        rates_list, ff_list = [], []

        # Temporal loop
        for step in range(self.N_STEPS):
            # update dynamics
            if self.LIVE_FF_UPDATE:
                ff_input, noise = live_ff_input(self, step, ff_input)
                if self.RATE_NOISE:
                    rates, rec_input = self.update_dynamics(
                        rates, ff_input, rec_input, Wab_T, W_stp_T
                    )
                    rates = rates + noise
                else:
                    rates, rec_input = self.update_dynamics(
                        rates, ff_input + noise, rec_input, Wab_T, W_stp_T
                    )
            else:
                if self.LR_TRAIN:
                    # ff_input = rl_ff_udpdate(self, ff_input, rates, step, self.RWD-1)
                    if self.IF_RL:
                        ff_input = rl_ff_udpdate(self, ff_input, rates, step, self.RWD)
                    else:
                        self.RWD = 22

                rates, rec_input = self.update_dynamics(
                    rates, ff_input[:, step], rec_input, Wab_T, W_stp_T
                )

            # update moving average
            mv_rates += rates
            if self.LIVE_FF_UPDATE and RET_FF:
                mv_ff += ff_input + noise

            # Reset moving average to start at 0
            if step == self.N_STEADY - self.N_WINDOW - 1:
                mv_rates = 0.0
                mv_ff = 0.0

            # update output every N_WINDOW steps
            if step >= self.N_STEADY:
                if step % self.N_WINDOW == 0:
                    if self.VERBOSE:
                        print_activity(self, step, rates)

                    if not REC_LAST_ONLY:
                        rates_list.append(mv_rates[..., self.slices[0]] / self.N_WINDOW)
                        if self.LIVE_FF_UPDATE and RET_FF:
                            ff_list.append(mv_ff[..., self.slices[0]] / self.N_WINDOW)
                        if self.IF_STP and RET_STP:
                            self.x_list.append(self.stp.x_stp)
                            self.u_list.append(self.stp.u_stp)

                    # Reset moving average
                    mv_rates = 0
                    mv_ff = 0

        # returns last step
        rates = rates[..., self.slices[0]]

        # returns full sequence
        if REC_LAST_ONLY == 0:
            # Stack list on 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
            rates = torch.stack(rates_list, dim=1)
            # del rates_list

            if self.LIVE_FF_UPDATE and RET_FF:  # returns ff input
                self.ff_input = torch.stack(ff_list, dim=1)
                # del ff_list

            if self.IF_STP and RET_STP:  # returns stp u and x
                self.u_list = torch.stack(self.u_list, dim=1)
                self.x_list = torch.stack(self.x_list, dim=1)

        # Add Linear readout (N_BATCH, N_EVAL_WIN, 1) on last few steps
        # if self.LR_READOUT:
        #     y_pred = self.low_rank.linear(self.low_rank.dropout(rates))
        #     # del rates
        #     return y_pred.squeeze(-1)

        if self.LIVE_FF_UPDATE == 0 and RET_FF:
            self.ff_input = ff_input[..., self.slices[0]]

        # del ff_input, rec_input

        # clear_cache()

        return rates
