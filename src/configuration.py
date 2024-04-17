from yaml import safe_load
import numpy as np

import torch
from torch.distributions import MultivariateNormal

from src.utils import set_seed


class Configuration:
    def __init__(self, conf_name, repo_root):
        self.conf_file = repo_root + "/conf/" + conf_name
        self.defaults = repo_root + "/conf/defaults.yml"

    def forward(self, **kwargs):
        parameters = safe_load(open(self.defaults, "r"))
        config = safe_load(open(self.conf_file, "r"))
        parameters.update(config)
        parameters.update(kwargs)

        self.__dict__.update(parameters)

        if self.FLOAT_PRECISION == 32:
            self.FLOAT = torch.float
        if self.FLOAT_PRECISION == 16:
            self.FLOAT = torch.float16
        if self.FLOAT_PRECISION == '16b':
            self.FLOAT = torch.bfloat16
        else:
            self.FLOAT = torch.float64

        self.device = torch.device(self.DEVICE)
        torch.set_default_dtype(self.FLOAT)
        # Set seed for the connectivity/input vectors
        set_seed(self.SEED)

        # create networks' constants
        init_const(self)

        return self

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def init_const(model):
    ##########################################
    # creating time steps from continuous time
    ##########################################
    model.N_STEADY = int(model.T_STEADY / model.DT)
    model.N_WINDOW = int(model.T_WINDOW / model.DT)
    model.N_STEPS = int(model.DURATION / model.DT) + model.N_STEADY + model.N_WINDOW

    model.N_STIM_ON = np.array(
        [int(i / model.DT) + model.N_STEADY for i in model.T_STIM_ON]
    )
    model.N_STIM_OFF = [int(i / model.DT) + model.N_STEADY for i in model.T_STIM_OFF]

    ##########################################
    # defining N and K per population
    ##########################################
    model.Na = []
    model.Ka = []

    # K is set to 1 in all2all nets
    if "all2all" in model.CON_TYPE:
        model.K = 1.0

    for i_pop in range(model.N_POP):
        model.Na.append(int(model.N_NEURON * model.frac[i_pop]))
        # model.Ka.append(model.K * const.frac[i_pop])
        model.Ka.append(model.K)

    model.Na = torch.tensor(model.Na, dtype=torch.int, device=model.device)
    model.Ka = torch.tensor(model.Ka, device=model.device)
    model.csumNa = torch.cat(
        (torch.tensor([0], device=model.device), torch.cumsum(model.Na, dim=0))
    )

    # slices[i] contains neurons' idx for population i
    model.slices = []
    for i_pop in range(model.N_POP):
        model.slices.append(slice(model.csumNa[i_pop], model.csumNa[i_pop + 1]))

    if model.VERBOSE:
        print("Na", model.Na, "Ka", model.Ka, "csumNa", model.csumNa)

    ##########################################
    # defining integration constants
    ##########################################

    # rate dynamics
    model.TAU = torch.tensor(model.TAU, device=model.device)
    model.EXP_DT_TAU = torch.ones(model.N_NEURON, device=model.device)
    model.DT_TAU = torch.ones(model.N_NEURON, device=model.device)

    for i_pop in range(model.N_POP):
        model.EXP_DT_TAU[model.slices[i_pop]] = torch.exp(-model.DT / model.TAU[i_pop])
        model.DT_TAU[model.slices[i_pop]] = model.DT / model.TAU[i_pop]

    model.THRESH = torch.tensor(model.THRESH, device=model.device)

    # synaptic dynamics
    model.TAU_SYN = torch.tensor(model.TAU_SYN, device=model.device)
    model.EXP_DT_TAU_SYN = torch.ones(model.N_NEURON, device=model.device)
    model.DT_TAU_SYN = torch.ones(model.N_NEURON, device=model.device)

    for i_pop in range(model.N_POP):
        model.EXP_DT_TAU_SYN[model.slices[i_pop]] = torch.exp(
            -model.DT / model.TAU_SYN[i_pop]
        )
        model.DT_TAU_SYN[model.slices[i_pop]] = model.DT / model.TAU_SYN[i_pop]

    # NMDA dynamics
    if model.IF_NMDA:
        model.TAU_NMDA = torch.tensor(model.TAU_NMDA, device=model.device)
        model.EXP_DT_TAU_NMDA = torch.ones(model.N_NEURON, device=model.device)
        model.DT_TAU_NMDA = torch.ones(model.N_NEURON, device=model.device)

        for i_pop in range(model.N_POP):
            model.EXP_DT_TAU_NMDA[model.slices[i_pop]] = torch.exp(
                -model.DT / model.TAU_NMDA[i_pop]
            )
            model.DT_TAU_NMDA[model.slices[i_pop]] = model.DT / model.TAU_NMDA[i_pop]

    ##########################################
    # defining connectivity constants
    ##########################################
    model.PROBA_TYPE = np.array(model.PROBA_TYPE).reshape(model.N_POP, model.N_POP)
    model.SIGMA = torch.tensor(model.SIGMA, device=model.device).view(
        model.N_POP, model.N_POP
    )
    model.KAPPA = torch.tensor(model.KAPPA, device=model.device).view(
        model.N_POP, model.N_POP
    )
    model.PHASE = torch.tensor(model.PHASE * torch.pi / 180.0, device=model.device)

    model.PHI0 = (
        torch.tensor(model.PHI0, device=model.device).unsqueeze(0) * torch.pi / 180.0
    )
    # model.PHI1 = torch.tensor(model.PHI1,  device=model.device).unsqueeze(0) * torch.pi / 180.0

    if "dual" in model.TASK:
        # if 'lr' in model.PROBA_TYPE[0, 0]:
        mean_ = torch.tensor(model.LR_MEAN, device=model.device, dtype=torch.float32)
        cov_ = torch.tensor(model.LR_COV, device=model.device, dtype=torch.float32)

        if cov_[0, 0] == cov_[0, 1]:
            # print('Using Hopfield like low rank')

            mean_ = mean_[[0, 2]]
            cov_ = torch.tensor(
                ([cov_[0, 0], cov_[0, 2]], [cov_[2, 0], cov_[2, 2]]),
                device=model.device, dtype=torch.float32
            )

            multivariate_normal = MultivariateNormal(mean_, cov_)
            model.PHI0 = multivariate_normal.sample((model.Na[0],)).T
            model.PHI0 = torch.stack(
                (model.PHI0[0], model.PHI0[0], model.PHI0[1], model.PHI0[1])
            ).type(model.float32)
        else:
            # print('Using Francesca like low rank')
            multivariate_normal = MultivariateNormal(mean_, cov_)
            model.PHI0 = multivariate_normal.sample((model.Na[0],)).T.type(model.FLOAT)

        del mean_, cov_
        del multivariate_normal
