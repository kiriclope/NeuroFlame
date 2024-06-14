import torch
import numpy as np

from src.stimuli import Stimuli
from src.lr_utils import get_theta


def live_ff_input(model, step, ff_input):
    noise = 0
    if model.VAR_FF[0, 0, 0] > 0:
        noise = torch.randn((model.N_BATCH, model.N_NEURON), device=model.device)
        for i_pop in range(model.N_POP):
            noise[:, model.slices[i_pop]].mul_(model.VAR_FF[:, i_pop])

    if step == 0:
        for i_pop in range(model.N_POP):
            if model.BUMP_SWITCH[i_pop]:
                ff_input[:, model.slices[i_pop]] = model.Ja0[:, i_pop] / torch.sqrt(
                    model.Ka[0]
                )
            else:
                ff_input[:, model.slices[i_pop]] = model.Ja0[:, i_pop]

    if step == model.N_STIM_ON[0]:
        for i_pop in range(model.N_POP):
            ff_input[:, model.slices[i_pop]] = model.Ja0[:, i_pop]

    if model.TASK != "None":
        if step in model.N_STIM_ON:
            i = np.where(model.N_STIM_ON == step)[0][0]

            size = (model.N_BATCH, model.Na[0])
            Stimulus = Stimuli(model.TASK, size, device=model.device)

            if "rand" in model.TASK:
                model.phase = (
                    torch.rand((size[0], 1), device=model.device) * 2.0 * torch.pi
                )

            theta = None
            if "dual" in model.TASK:
                theta = get_theta(model.PHI0[0], model.PHI0[2]).unsqueeze(0)

            if "rand" in model.TASK:
                Stimulus.task = "odr"
                stimulus = Stimulus(
                    model.I0[i], model.SIGMA0[i], model.phase, theta=theta
                )

            elif "dual" in model.TASK:
                if "odr" in model.TASK:
                    phase = torch.ones((size[0], 1), device=model.device)
                    phase = phase * model.PHI1[i]

                    if i == 0:
                        Stimulus.task = "dual"
                        stimulus = Stimulus(
                            model.I0[i], model.SIGMA0[i], model.PHI0[2 * i + 1]
                        )
                    else:
                        Stimulus.task = "odr"
                        stimulus = Stimulus(
                            model.I0[i], model.SIGMA0[i], phase, theta=theta
                        )
                else:
                    if model.LR_TRAIN:
                        if model.I0[i] > 0:
                            stimulus = Stimulus(
                                model.I0[i], model.SIGMA0[i], model.odors[i])
                        else:
                            stimulus = Stimulus(
                                model.I0[i], model.SIGMA0[i], model.odors[5+i])
                    else:
                        stimulus = Stimulus(
                            model.I0[i], model.SIGMA0[i], model.PHI0[2 * i + 1]
                        )

                        if i == 0:
                            # multiply last half of stimulus by -1 to get two samples A/B
                            stimulus = model.stim_mask[:, i] * stimulus
            else:
                stimulus = Stimulus(model.I0[i], model.SIGMA0[i], model.PHI0[:, i])

            ff_input[:, model.slices[0]] = (
                model.Ja0[:, 0] + torch.sqrt(model.Ka[0]) * model.M0 * stimulus
            )

        if step in model.N_STIM_OFF:
            ff_input[:, model.slices[0]] = model.Ja0[:, 0]

    return ff_input, noise


def init_ff_live(model):
    # Here, ff_input is (N_BATCH, N_NEURON) and is updated at each timestep.
    # Otherwise, ff_input is (N_BATCH, N_STEP, N_NEURON).
    # Live FF update is recommended when dealing with large batch size.

    model.stim_mask = torch.ones((model.N_BATCH, 2, model.Na[0]), device=model.device)

    model.stim_mask[model.N_BATCH // 2 :] = -1

    ff_input = torch.zeros((model.N_BATCH, model.N_NEURON), device=model.device)

    ff_input, _ = live_ff_input(model, 0, ff_input)

    return ff_input


def init_ff_seq(model):
    """
    Creates ff input to the network for all timesteps.
    Inputs can be noisy or not and depends on the task.
    returns:
    ff_input: tensorfloat of size (N_BATCH, N_STEPS, N_NEURON)
    """

    ff_input = torch.randn(
        (model.N_BATCH, model.N_STEPS, model.N_NEURON),
        device=model.device,
    )

    for i_pop in range(model.N_POP):
        ff_input[..., model.slices[i_pop]].mul_(model.VAR_FF[:, i_pop])

    for i_pop in range(model.N_POP):
        if model.BUMP_SWITCH[i_pop]:
            ff_input[:, : model.N_STIM_ON[0], model.slices[i_pop]].add_(
                model.Ja0[:, i_pop] / torch.sqrt(model.Ka[0])
            )
        else:
            ff_input[:, : model.N_STIM_ON[0], model.slices[i_pop]].add_(
                model.Ja0[:, i_pop]
            )

    for i_pop in range(model.N_POP):
        ff_input[:, model.N_STIM_ON[0] :, model.slices[i_pop]].add_(model.Ja0[:, i_pop])

    if model.TASK != "None":
        size = (model.N_BATCH, model.Na[0])
        Stimulus = Stimuli(model.TASK, size, device=model.device)

        if "rand" in model.TASK:
            model.phase = torch.rand((size[0], 1), device=model.device) * 2.0 * torch.pi

            theta = None
            if "dual" in model.TASK:
                if "rand" in model.TASK:
                    theta = get_theta(model.PHI0[0], model.PHI0[2]).unsqueeze(0)

        for i, _ in enumerate(model.N_STIM_ON):
            if "rand" in model.TASK:
                Stimulus.task = "odr"
                stimulus = Stimulus(
                    model.I0[i], model.SIGMA0[i], model.phase, theta=theta
                )
            elif "dual" in model.TASK:
                if model.LR_TRAIN:
                    if 0==0:
                    # if (model.IF_RL == 0 or i!=model.RWD): # or (model.IF_RL==0 and i!=model.RWD-1):
                        # stimulus = Stimulus(
                        #     model.I0[i], model.SIGMA0[i], model.odors[i])
                        if model.I0[i] > 0:
                            stimulus = Stimulus(
                                model.I0[i], model.SIGMA0[i], model.odors[i])
                        else:
                            stimulus = Stimulus(
                                model.I0[i], model.SIGMA0[i], model.odors[5+i])
                    else:
                        stimulus = 0
                else:
                    stimulus = Stimulus(
                        model.I0[i], model.SIGMA0[i], model.PHI0[2 * i + 1]
                    )
            else:
                stimulus = Stimulus(model.I0[i], model.SIGMA0[i], model.PHI0[:, i])

            # reshape stimulus to be (N_BATCH, 1, NE) adding dummy time dimension
            # stimulus = stimulus.unsqueeze(1)
            # print(stimulus.shape)

            if model.N_STIM_ON[i] < model.N_STEPS:
                ff_input[:, model.N_STIM_ON[i] : model.N_STIM_OFF[i], model.slices[0]].add_(
                    stimulus
                )
            del stimulus

    return ff_input * torch.sqrt(model.Ka[0]) * model.M0


def rl_ff_udpdate(model, ff_input, rates, step, rwd):
    if step == model.N_STIM_ON[rwd]:
        size = (model.N_BATCH, model.Na[0])

        # overlap = 1.0 * ( (rates[:, model.slices[0]] @ model.low_rank.linear.weight[0]) > 0).unsqueeze(-1).unsqueeze(-1)
        # overlap = 1.0 * ( (rates[:, model.slices[0]] @ model.low_rank.U[model.slices[0], 1]) > 0).unsqueeze(-1).unsqueeze(-1)

        if model.VERBOSE:
            print('overlap', overlap)

        Stimulus = Stimuli(model.TASK, size, device=model.device)
        # print('RWD', model.I0[rwd])
        # stimulus = Stimulus(model.I0[rwd], 1.0, model.low_rank.U[model.slices[0], 1])
        # stimulus = Stimulus(1.0, 1.0, model.low_rank.U[model.slices[0], 1])
        stimulus = Stimulus(model.I0[rwd], model.SIGMA0[rwd], model.low_rank.linear.weight[0])

        # print(rates.shape, overlap.shape, stimulus.shape, (overlap * stimulus).shape)

        # ff_input[:, model.N_STIM_ON[rwd] : model.N_STIM_OFF[rwd], model.slices[0]] = (
        #     ff_input[:, model.N_STIM_ON[rwd] : model.N_STIM_OFF[rwd], model.slices[0]]
        #     + overlap * stimulus * torch.sqrt(model.Ka[0]) * model.M0
        # )

        # if rwd == model.RWD:
        #     ff_input[:, model.N_STIM_ON[rwd] : model.N_STIM_OFF[rwd], model.slices[0]] = (
        #         ff_input[:, model.N_STIM_ON[rwd] : model.N_STIM_OFF[rwd], model.slices[0]]
        #         + overlap * stimulus * torch.sqrt(model.Ka[0]) * model.M0
        #     )
        # else:
        ff_input[:, model.N_STIM_ON[rwd] : model.N_STIM_OFF[rwd], model.slices[0]] = (
            ff_input[:, model.N_STIM_ON[rwd] : model.N_STIM_OFF[rwd], model.slices[0]]
            + stimulus * torch.sqrt(model.Ka[0]) * model.M0
        )

    return ff_input

def init_ff_input(model):
    if model.LIVE_FF_UPDATE:
        return init_ff_live(model)
    return init_ff_seq(model)
