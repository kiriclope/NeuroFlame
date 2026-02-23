import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True

from src.configuration import Configuration
from src.connectivity import Connectivity
from src.activation import Activation
from src.plasticity import Plasticity
from src.hebbian import Hebbian
from src.lr_utils import LowRankWeights, clamp_tensor, normalize_tensor

from src.ff_input import live_ff_input, init_ff_input
from src.utils import set_seed, clear_cache, print_activity

import warnings

warnings.filterwarnings("ignore")

def flex_matmul(a, b):
    # If a is 2D and b is 3D: unsqueeze, matmul, then squeeze
    if (a.dim() == 2) and (b.dim() == 3):
        res = torch.matmul(a.unsqueeze(1), b)
        return res.squeeze(1)
    # If both are 2D: just matmul
    elif (a.dim() == 2) and (b.dim() == 2):
        return torch.matmul(a, b)
    # (Extend with other logic if needed)
    else:
        raise ValueError("Unsupported tensor dimensions: {}, {}".format(a.shape, b.shape))


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
        self.initTrainWeights()

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

    def initTrainWeights(self):
        shape = (self.Na[0], self.Na[0])
        if self.TRAIN_EI:
            shape = (self.N_NEURON, self.N_NEURON)
            self.train_mask = torch.zeros(shape, device=self.device)

        self.Wab_train = nn.Parameter(torch.randn(shape, device=self.device) * self.LR_INI)

        if self.TRAIN_EI:
            self.train_mask = torch.zeros(shape, device=self.device)
            for i_pop in range(self.N_POP):
                for j_pop in range(self.N_POP):
                    self.train_mask[self.slices[i_pop], self.slices[j_pop]] = self.IS_TRAIN[i_pop][j_pop]

    def initWeights(self):
        """
        Initializes the connectivity matrix self.Wab.
        Loops over blocks to create the full matrix.
        Relies on class Connectivity from connectivity.py
        """

        # in pytorch, Wij is i to j.
        self.Wab_T = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)

        # Creates connetivity matrix in blocks
        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):
                weight_mat = Connectivity(self.Na[i_pop], self.Na[j_pop], self.Ka[j_pop], device=self.device)

                weights = weight_mat(self.CON_TYPE, self.PROBA_TYPE[i_pop][j_pop],
                                     kappa=self.KAPPA[i_pop][j_pop],
                                     phase=self.PHASE,
                                     sigma=self.SIGMA[i_pop][j_pop],
                                     lr_mean=self.LR_MEAN,
                                     lr_cov=self.LR_COV,
                                     ksi=self.PHI0,
                                     )

                self.Wab_T.data[self.slices[i_pop], self.slices[j_pop]] = self.Jab[i_pop][j_pop] * weights

        self.Wab_T = self.Wab_T.T
        # self.register_buffer('Wab_T', self.Wab_T)

    def initSTP(self):
        """Creates short term plasticity weights."""

        self.J_STP = torch.tensor(self.J_STP, device=self.device)
        if self.training:
            self.J_STP = nn.Parameter(self.J_STP)

        # NEED .clone() here otherwise BAD THINGS HAPPEN !!!
        # this for LR RNN
        # self.W_stp_T = [self.GAIN * self.Wab_T[self.slices[0], self.slices[0]].clone()
        #                 / self.Jab[0, 0]
        #                 / torch.sqrt(self.Ka[0])]

        k = 0
        self.W_stp_T = []
        for i in range(self.N_POP): # post
            post_slice = self.slices[i]
            for j in range(self.N_POP): # pre
                pre_slice = self.slices[j]
                if self.IS_STP[j+i*self.N_POP]:
                    block = self.Wab_T[self.slices[j], self.slices[i]].clone() / torch.abs(self.Jab[i, j])
                    # / torch.sqrt(self.Ka[j])
                    self.W_stp_T.append(block)

                    # remove non-plastic part from Wab_T
                    with torch.no_grad():
                        self.Wab_T[pre_slice, post_slice].zero_()

                    if self.TRAIN_EI:
                        self.train_mask[pre_slice, post_slice] = 0.0

                    if self.LR_TYPE == 'full':
                        self.W_stp_T[k] = 1.0 / self.Na[j]

                    k = k + 1

    def forwardSTP(self, IF_INIT=0):
        self.stp = []
        # Need this here otherwise autograd complains

        if IF_INIT:
            self.u_stp_last = []
            self.x_stp_last = []

        k = 0
        for i in range(self.N_POP): # post
            for j in range(self.N_POP): # pre
                if self.IS_STP[j+i*self.N_POP]:
                    self.stp.append(Plasticity(self.USE[j+i*self.N_POP],
                                               self.TAU_FAC[j+i*self.N_POP],
                                               self.TAU_REC[j+i*self.N_POP],
                                               self.DT,
                                               (self.N_BATCH, self.Na[j]),
                                               STP_TYPE=self.STP_TYPE,
                                               IF_INIT=IF_INIT,
                                               device=self.device,
                                               ))

                    # previous state is loaded
                    if IF_INIT:
                        self.u_stp_last.append(torch.zeros_like(self.stp[k].u_stp.detach()))
                        self.x_stp_last.append(torch.zeros_like(self.stp[k].x_stp.detach()))
                    else:
                        # if IF_INIT==0:
                        self.stp[k].u_stp = self.u_stp_last[k].detach()
                        self.stp[k].x_stp = self.x_stp_last[k].detach()

                    k = k + 1

        self.x_list, self.u_list = [], []

        return self


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


    def update_dynamics(self, rates, ff_input, ff_prev, rec_input, Wab_T, W_stp_T, thresh):
        """Updates the dynamics of the model at each timestep"""

        hidden = flex_matmul(rates, Wab_T) # here @ is sum_j rj * Wji hence it's j to i (row are presyn)

        # update stp variables
        if self.IF_STP:
            k=0
            for i in range(self.N_POP): # post
                for j in range(self.N_POP): # pre
                    if self.IS_STP[j+i*self.N_POP]:
                        Aux = self.stp[k](rates[:, self.slices[j]]) # pre
                        hidden_stp = Aux @ W_stp_T[k]
                        hidden[:, self.slices[i]] = hidden[:, self.slices[i]] + hidden_stp # post
                        k = k + 1

        if self.IF_FF_STP:
            hidden_ff_stp = torch.sign(ff_input) * (self.ff_stp(torch.abs(ff_input) / torch.sqrt(self.Ka[0]))
                                                    * torch.sqrt(self.Ka[0]))
            ff_input = ff_input + hidden_ff_stp

        if self.IF_FF_DYN:
            ff_input = ff_prev * self.EXP_FF + ff_input * (1.0 - self.EXP_FF)
            ff_prev = ff_input

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
            ff_input = torch.sign(ff_input) * nn.ReLU()(ff_input - self.thresh_ff)
            self.thresh_ff = self.thresh_ff * self.EXP_FF_ADAPT + nn.ReLU()(ff_input) * self.A_FF_ADAPT * (1.0-self.EXP_FF_ADAPT)

        # compute net input
        net_input = ff_input + rec_input[0]

        if self.IF_NMDA:
            if Wab_T.dim()==2:
                hidden = flex_matmul(rates[:, self.slices[0]], Wab_T[self.slices[0]])
            else:
                hidden = flex_matmul(rates[:, self.slices[0]], Wab_T[:, self.slices[0]])

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

        # adaptation
        if self.IF_ADAPT:
            thresh[:, self.slices[0]] = thresh[:, self.slices[0]] * self.EXP_ADAPT + rates[:, self.slices[0]].detach() * self.A_ADAPT * (1.0 - self.EXP_ADAPT)

        return rates, ff_prev, rec_input, thresh


    def forwardFFSTP(self, IF_INIT):
        self.ff_stp = Plasticity(self.FF_USE,self.TAU_FF_FAC, self.TAU_FF_REC, self.DT,
                                 (self.N_BATCH, self.N_NEURON),
                                 STP_TYPE=self.STP_TYPE,
                                 IF_INIT=IF_INIT,
                                 device=self.device,
                                 )

        if IF_INIT:
            self.u_ff_stp_last = torch.zeros_like(self.ff_stp.u_stp)
            self.x_ff_stp_last = torch.zeros_like(self.ff_stp.x_stp)
        else:
            # if IF_INIT==0:
            # # previous state is loaded
            self.ff_stp.u_stp = self.u_ff_stp_last
            self.ff_stp.x_stp = self.x_ff_stp_last

        return self

    def forwardHebb(self, hebb_rates, rates, Wab_T):

        for j in range(self.N_POP): # pre
            pre = rates[:, self.slices[j]]
            for i in range(self.N_POP): # post
                post = rates[:, self.slices[i]]
                if self.IS_HEBB[j+i*self.N_POP]:
                    W_hebb = self.hebb(pre, post, hebb_rates[:, self.slices[j]], hebb_rates[:, self.slices[i]])
                    W_hebb /= torch.sqrt(self.Ka[0])

                    # W_hebb = W_hebb.transpose(1, 2) / torch.sqrt(self.Ka[0])
                    # if j==0:
                    #     W_hebb = W_hebb.clamp(min=0.0)
                    # else:
                    #     W_hebb = W_hebb.clamp(max=0.0)

                    Wab_T[:, self.slices[j], self.slices[i]] = self.Wab_T[self.slices[j], self.slices[i]].unsqueeze(0)
                    Wab_T[:, self.slices[j], self.slices[i]] += W_hebb

        Wab_T[:, self.slices[0]] = Wab_T[:, self.slices[0]].clamp(min=0.0)
        Wab_T[:, self.slices[1]] = Wab_T[:, self.slices[1]].clamp(max=0.0)

        # Wab_T = self.Wab_T.unsqueeze(0) + Wab_T_train

        if self.HEBB_TYPE=='bcm':
            hebb_rates = hebb_rates * self.EXP_HEBB + rates * (1.0 - self.EXP_HEBB)

        return Wab_T

    def save_last_step(self, step, rates, rec_input, hebb_rates):
        end_idx = torch.where(step==self.end_indices[-1])[0]

        self.end_mask[end_idx, 0] = float('nan')
        self.rates_last[end_idx] = rates[end_idx].detach()

        self.rec_input_last[:, end_idx] = rec_input[:, end_idx].detach()
        self.thresh_last[end_idx] = thresh[end_idx].detach()

        if self.IF_HEBB:
            self.hebb_rates_last[end_idx] = hebb_rates[end_idx].detach()

        if self.IF_STP:
            k = 0
            for i in range(self.N_POP):
                for j in range(self.N_POP):
                    if self.IS_STP[j+i*self.N_POP]:
                        self.u_stp_last[k][end_idx] = self.stp[k].u_stp[end_idx].detach()
                        self.x_stp_last[k][end_idx] = self.stp[k].x_stp[end_idx].detach()
                        k = k + 1

        if self.IF_FF_STP:
            # self.u_ff_stp_last[end_idx] = self.ff_stp.u_stp[end_idx]
            self.x_ff_stp_last[end_idx] = self.ff_stp.x_stp[end_idx].detach()


    def forward(self, ff_input=None, RET_STP=0, RET_REC=0, IF_INIT=1):
        """
        Main method of Network class, runs networks dynamics over set of timesteps
        and returns rates at each time point or just the last time point.
        args:
        :param ff_input: float (N_BATCH, N_STEP, N_NEURONS), ff inputs into the network.
        rates_list:
        :param rates_list: float (N_BATCH, N_STEP or 1, N_NEURONS), rates of the neurons.
        """

        # Initialization (if  ff_input is None, ff_input is generated)
        if IF_INIT:
            rates, ff_input, rec_input, thresh = self.initRates(ff_input)

            if self.TRAINING == 0:
                with torch.no_grad():
                    self.rates_last = torch.zeros_like(rates.detach())
                    self.rec_input_last = torch.zeros_like(rec_input.detach())
                    self.thresh_last = torch.zeros_like(thresh.detach())
                    self.end_mask = torch.ones((self.N_BATCH, 1)).to(self.device)

        else:
            rates, rec_input, thresh = self.rates_last, self.rec_input_last, self.thresh_last

        # Add STP
        if self.IF_STP:
            self.forwardSTP(IF_INIT)

        if self.IF_FF_STP:
            self.forwardFFSTP(IF_INIT)

        Wab_T = self.Wab_T

        if self.LR_TRAIN:
            self.Wab_train = self.low_rank(self.LR_NORM, self.LR_CLAMP)

        if self.IF_HEBB:
            hebb_rates = rates.clone()
            self.hebb = Hebbian(self.ETA, self.DT, self.HEBB_TYPE, self.HEBB_FRAC)
            Wab_T = Wab_T.unsqueeze(0).repeat(self.N_BATCH, 1, 1) # (N_BATCH, N_NEURON, N_NEURON)

            if IF_INIT:
                self.hebb_rates_last = torch.zeros_like(hebb_rates.detach())
            else:
                hebb_rates = self.hebb_rates_last

        W_stp_T = None
        if self.IF_STP:
            # this for lr rnn
            # W_stp_T = [self.GAIN * self.J_STP * (self.W_stp_T[0] + self.Wab_train / self.Na[0])]
            # need to keep the scale for odr rnn's
            # W_stp_T = [self.GAIN * self.J_STP * (self.W_stp_T[0]
            #                                      + self.Wab_train[self.slices[0], self.slices[0]]) / torch.sqrt(self.Ka[0])]

            W_stp_T = [self.GAIN * self.J_STP * (1.0 + self.Wab_train[self.slices[0], self.slices[0]]) / self.Na[0]]

            # W_stp_T = [self.GAIN * self.J_STP * self.Wab_train[self.slices[0], self.slices[0]] / self.Na[0]]

            if self.CLAMP:
                W_stp_T[0] = clamp_tensor(W_stp_T[0], 0, self.slices)

            k = 1
            for i in range(self.N_POP): # post
                for j in range(self.N_POP): # pre
                    if (self.IS_STP[j+i*self.N_POP]) and ((i+j)!=0):
                        W_stp_T.append(self.GAIN * self.W_STP[j+i*self.N_POP]
                                       * self.J_STP * self.W_stp_T[k] / torch.sqrt(self.Ka[j]))
                        k = k + 1

        if self.TRAIN_EI:
            Wab_train = normalize_tensor(self.Wab_train, 0, self.slices, self.Na)
            Wab_train = normalize_tensor(Wab_train, 1, self.slices, self.Na)

            Wab_T = self.GAIN * (self.Wab_T + self.train_mask * Wab_train)

            if self.CLAMP:
                # Check indices Think need some transpose
                Wab_T = clamp_tensor(Wab_T.T, 0, self.slices).T
                Wab_T = clamp_tensor(Wab_T.T, 1, self.slices).T

        if self.IF_OPTO:
            rand_idx = torch.randperm(W_stp_T[0].size(0))[:self.N_OPTO]
            W_stp_T[0][rand_idx] = 0

            # _, idx = torch.sort(self.low_rank.V[:,1])
            # W_stp_T[idx[:self.N_OPTO]] = 0

        # Moving average
        ff_prev = 0
        mv_rates, mv_rec = 0, 0
        self.rates_list, self.rec_list = [], []

        # Temporal loop
        for step in range(self.N_STEPS):
            # hebbian learning
            if self.IF_HEBB and (step>=self.N_HEBB):
                self.forwardHebb(hebb_rates, rates, Wab_T)

            # add noise to the rates
            if self.RATE_NOISE:
                rate_noise = torch.randn((self.N_BATCH, self.N_NEURON), device=self.device)
                rates = rates + rate_noise * self.VAR_RATE

            ff_step = live_ff_input(self, step, ff_input) if self.LIVE_FF_UPDATE else ff_input[:, step]
            rates, ff_prev, rec_input, thresh = self.update_dynamics(rates, ff_step, ff_prev, rec_input, Wab_T, W_stp_T, thresh)

            # save last state
            if (self.TRAINING==0) and torch.any(step==self.end_indices[-1]):
                self.save_last_step(step, rates, rec_input, hebb_rates)

            # update moving average
            if step >= (self.N_STEADY + self.N_HEBB - self.N_WINDOW - 1):
                mv_rates += rates
                if RET_REC:
                    mv_rec += rec_input

            # update output every N_WINDOW steps
            if (step >= (self.N_STEADY + self.N_HEBB)) and (step % self.N_WINDOW == 0):
                if self.VERBOSE:
                    print_activity(self, step, rates)

                self.rates_list.append(mv_rates[:, self.slices[0]] / self.N_WINDOW)
                if RET_REC:
                    self.rec_list.append(mv_rec[..., self.slices[0]] / self.N_WINDOW)

                if self.IF_STP and RET_STP:
                    self.x_list.append(self.stp[0].x_stp)
                    self.u_list.append(self.stp[0].u_stp)

                mv_rates, mv_rec = 0, 0

        if self.IF_HEBB:
            self.W_hebb_T = Wab_T

        # Stack list on 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
        rates = torch.stack(self.rates_list, dim=1)

        if RET_REC:
            self.rec_input = torch.stack(self.rec_list, dim=2)

        if self.IF_STP and RET_STP:
            self.u_list = torch.stack(self.u_list, dim=1)
            self.x_list = torch.stack(self.x_list, dim=1)

        if self.LR_TRAIN:
            self.readout = rates @ self.low_rank.V[self.slices[0]] / self.Na[0]
            if self.LR_READOUT==1:
                linear = self.low_rank.linear(self.dropout(rates)) / self.Na[0]
                self.readout = torch.cat((self.readout, linear), dim=-1)

        # clear_cache()

        return rates
