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
            torch.set_float32_matmul_precision('medium')
        elif self.FLOAT_PRECISION == 16:
            self.FLOAT = torch.float16
        elif self.FLOAT_PRECISION == "16b":
            self.FLOAT = torch.bfloat16
        else:
            self.FLOAT = torch.float64

        self.device = torch.device(self.DEVICE)
        torch.set_default_dtype(self.FLOAT)
        # Set seed for the connectivity/input vectors
        set_seed(self.SEED)

        # create networks' constants
        init_time_const(self)
        init_const(self)

        return self

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def init_time_const(model):
    ##########################################
    # creating time steps from continuous time
    ##########################################
    model.N_STEADY = int(model.T_STEADY / model.DT)
    model.N_WINDOW = int(model.T_WINDOW / model.DT)
    model.N_STEPS = int(model.DURATION / model.DT) + model.N_WINDOW + model.N_STEADY

    model.N_STIM_ON = torch.tensor([int(i / model.DT) + model.N_STEADY for i in model.T_STIM_ON]).to(model.device)
    model.N_STIM_OFF = torch.tensor([int(i / model.DT) + model.N_STEADY for i in model.T_STIM_OFF]).to(model.device)

    model.random_shifts = torch.zeros((model.N_BATCH,)).to(model.device)
    model.start_indices = (model.N_STIM_ON.unsqueeze(-1) + model.random_shifts)
    model.end_indices = (model.N_STIM_OFF.unsqueeze(-1) + model.random_shifts)

    if model.RANDOM_DELAY:
        N_MAX_DELAY = model.MAX_DELAY / model.DT
        N_MIN_DELAY = model.MIN_DELAY / model.DT

        model.N_STEPS += int(N_MAX_DELAY - N_MIN_DELAY)

        model.random_shifts = torch.randint(low=int(N_MIN_DELAY), high=int(N_MAX_DELAY), size=(model.N_BATCH,)).to(model.device)

        model.start_indices = (model.N_STIM_ON.unsqueeze(-1) + model.random_shifts)
        model.end_indices = (model.N_STIM_OFF.unsqueeze(-1) + model.random_shifts)

        if 'odr' in model.TASK:
            model.start_indices[0] = model.N_STIM_ON[0]
            model.end_indices[0] = model.N_STIM_OFF[0]

        if 'dual' in model.TASK:
            # random DPA delay
            model.start_indices[:-1] = model.N_STIM_ON[:-1]
            model.end_indices[:-1] = model.N_STIM_OFF[:-1]

def init_const(model):
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
        if model.FRAC_K:
            model.Ka.append(model.K * model.frac[i_pop])
        else:
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

    # Adaptation
    model.THRESH = torch.tensor(model.THRESH, device=model.device)
    model.thresh = torch.ones(model.N_BATCH, model.N_NEURON, device=model.device)

    for i_pop in range(model.N_POP):
        model.thresh[:, model.slices[i_pop]] = model.THRESH[i_pop]

    if model.IF_FF_ADAPT:
        model.TAU_FF_ADAPT = torch.tensor(model.TAU_FF_ADAPT, device=model.device)
        model.EXP_FF_ADAPT = torch.exp(-model.DT / model.TAU_FF_ADAPT)

    if model.IF_ADAPT:
        model.TAU_ADAPT = torch.tensor(model.TAU_ADAPT, device=model.device)
        model.EXP_DT_TAU_ADAPT = torch.ones(model.N_NEURON, device=model.device)
        model.DT_TAU_ADAPT = torch.ones(model.N_NEURON, device=model.device)

        for i_pop in range(model.N_POP):
            model.EXP_DT_TAU_ADAPT[model.slices[i_pop]] = torch.exp(-model.DT / model.TAU_ADAPT[i_pop])
            model.DT_TAU_ADAPT[model.slices[i_pop]] = model.DT / model.TAU_ADAPT[i_pop]

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
    # defining models' Parameters
    ##########################################
    model.Jab = (
        torch.tensor(model.Jab, device=model.device).reshape(model.N_POP, model.N_POP)
    )

    for i_pop in range(model.N_POP):
        model.Jab[:, i_pop] = model.Jab[:, i_pop] / torch.sqrt(model.Ka[i_pop])

    model.Ja0 = torch.tensor(model.Ja0, device=model.device)
    # now inputs are scaled in init_ff_input unless live update

    model.Ja0 = model.Ja0.unsqueeze(0)  # add batch dim
    model.Ja0 = model.Ja0.unsqueeze(-1)  # add neural dim

    model.VAR_FF = torch.sqrt(torch.tensor(model.VAR_FF, device=model.device))
    model.VAR_RATE = torch.sqrt(torch.tensor(model.VAR_RATE, device=model.device)) / torch.sqrt(model.Ka[0])
    # scaling ff variance as O(1) because we multiply by sqrtK in seq input
    # model.VAR_FF.mul_(1.0 / torch.sqrt(model.Ka[0]))

    model.VAR_FF.mul_(model.M0)
    model.VAR_FF = model.VAR_FF.unsqueeze(0)  # add batch dim
    model.VAR_FF = model.VAR_FF.unsqueeze(-1)  # add neural dim

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

    if isinstance(model.PHI0, list):
        model.PHI0 = torch.tensor(model.PHI0, device=model.device).unsqueeze(0)

    # model.PHI0 = model.PHI0 * torch.pi / 180.0
    # model.PHI1 = torch.tensor(model.PHI1,  device=model.device).unsqueeze(0) * torch.pi / 180.0

    model.IS_TRAIN = torch.tensor(model.IS_TRAIN, device=model.device).view(
        model.N_POP, model.N_POP
    )

    if "dual" in model.TASK:
        # if 'lr' in model.PROBA_TYPE[0, 0]:
        mean_ = torch.tensor(model.LR_MEAN, device=model.device, dtype=torch.float32)
        cov_ = torch.tensor(model.LR_COV, device=model.device, dtype=torch.float32)

        if cov_[0, 0] == cov_[0, 1]:
            # print('Using Hopfield like low rank')

            mean_ = mean_[[0, 2]]
            cov_ = torch.tensor(
                ([cov_[0, 0], cov_[0, 2]], [cov_[2, 0], cov_[2, 2]]),
                device=model.device,
                dtype=torch.float32,
            )

            multivariate_normal = MultivariateNormal(mean_, cov_)
            model.PHI0 = multivariate_normal.sample((model.Na[0],)).T
            model.PHI0 = torch.stack(
                (model.PHI0[0], model.PHI0[0], model.PHI0[1], model.PHI0[1])
            ).type(model.FLOAT)
        else:
            # print('Using Francesca like low rank')
            multivariate_normal = MultivariateNormal(mean_, cov_)
            model.PHI0 = multivariate_normal.sample((model.Na[0],)).T.type(model.FLOAT)

        del mean_, cov_
        del multivariate_normal
