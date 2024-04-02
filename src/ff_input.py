import torch
import numpy as np

from src.stimuli import Stimuli
from src.lr_utils import get_theta

# class Feedforward():
#     def __init__(self, LIVE_FF_UPDATE):
#         a = 0
    
def live_ff_input(model, step, ff_input):

    noise = 0
    if model.VAR_FF[0, 0, 0]>0:
        noise = torch.randn((model.N_BATCH, model.N_NEURON), dtype=model.FLOAT, device=model.device)
        for i_pop in range(model.N_POP):
            noise[:, model.slices[i_pop]].mul_(model.VAR_FF[:, i_pop])

    if step==0:
        for i_pop in range(model.N_POP):
            if model.BUMP_SWITCH[i_pop]:
                ff_input[:, model.slices[i_pop]] = model.Ja0[:, i_pop] / torch.sqrt(model.Ka[0])
            else:
                ff_input[:, model.slices[i_pop]] = model.Ja0[:, i_pop]

    if step==model.N_STIM_ON[0]:
        for i_pop in range(model.N_POP):
            ff_input[:, model.slices[i_pop]] = model.Ja0[:, i_pop]

    stimulus = 0

    if model.TASK != 'None':
        if step in model.N_STIM_ON:
            i = np.where(model.N_STIM_ON == step)[0][0]

            size = (model.N_BATCH, model.Na[0])

            if 'dual' in model.TASK:
                if 'rand' in model.TASK:
                    theta = get_theta(model.PHI0[0], model.PHI0[2])
                    if i==0:
                        phase = torch.rand(size[0], dtype=model.FLOAT, device=model.device) * 360
                        model.phase = phase.unsqueeze(1).expand((phase.shape[0], size[-1]))

                    stimulus = Stimuli('odr', size, device=model.device)(model.I0[i],
                                                                        model.SIGMA0[i],
                                                                        model.phase,
                                                                        rnd_phase=0,
                                                                        theta_list=theta)
                    del theta

                elif 'odr' in model.TASK:
                    theta = get_theta(model.PHI0[0], model.PHI0[2])
                    phase = torch.ones(size[0], device=model.device)
                    phase = phase.unsqueeze(1).expand((phase.shape[0], size[-1])) * model.PHI1[i]

                    if i == 0:
                        stimulus = Stimuli('dual', size)(model.I0[i],
                                                            model.SIGMA0[i],
                                                            model.PHI0[2*i+1])

                        stimulus = model.stim_mask[:, i] * stimulus

                    # if i == 0:
                    #     phase = phase + model.stim_mask[:, i] * 180
                    else:
                        stimulus = Stimuli('odr', size, device=model.device)(model.I0[i],
                                                                            model.SIGMA0[i],
                                                                            phase,
                                                                            rnd_phase=0,
                                                                            theta_list=theta)
                    del theta, phase
                else:
                    stimulus = Stimuli(model.TASK, size)(model.I0[i],
                                                        model.SIGMA0[i],
                                                        model.PHI0[2*i+1])
                    if i == 0:
                        stimulus = model.stim_mask[:, i] * stimulus
            else:
                rnd_phase = 0
                if 'rand' in model.TASK:
                    rnd_phase = 1

                
                stimulus = Stimuli(model.TASK, size)(model.I0[i],
                                                    model.SIGMA0[i],
                                                    model.PHI0[:, i],
                                                    rnd_phase=rnd_phase)

            ff_input[:, model.slices[0]] = model.Ja0[:, 0] + torch.sqrt(model.Ka[0]) * model.M0 * stimulus
            # del stimulus

        if step in model.N_STIM_OFF:
            ff_input[:, model.slices[0]] = model.Ja0[:, 0]

    return ff_input, noise

def init_ff_live(model):
    
    # Here, ff_input is (N_BATCH, N_NEURON) and is updated at each timestep.
    # Otherwise, ff_input is (N_BATCH, N_STEP, N_NEURON).
    # Live FF update is recommended when dealing with large batch size.

    model.stim_mask = torch.ones((model.N_BATCH, 2, model.Na[0]),
                                dtype=model.FLOAT, device=model.device)

    model.stim_mask[model.N_BATCH//2:] = -1

    ff_input = torch.zeros((model.N_BATCH, model.N_NEURON),
                            dtype=model.FLOAT, device=model.device)
    
    ff_input, _ = live_ff_input(model, 0, ff_input)

    return ff_input

def init_ff_seq(model):
    """
    Creates ff input to the network for all timesteps.
    Inputs can be noisy or not and depends on the task.
    returns:
    ff_input: tensorfloat of size (N_BATCH, N_STEPS, N_NEURON)
    """

    ff_input = torch.randn((model.N_BATCH, model.N_STEPS, model.N_NEURON), dtype=model.FLOAT, device=model.device)

    for i_pop in range(model.N_POP):
        ff_input[..., model.slices[i_pop]].mul_(model.VAR_FF[:, i_pop])

    for i_pop in range(model.N_POP):
        if model.BUMP_SWITCH[i_pop]:
            ff_input[:, :model.N_STIM_ON[0], model.slices[i_pop]].add_(model.Ja0[:, i_pop] / torch.sqrt(model.Ka[0]))
        else:
            ff_input[:, :model.N_STIM_ON[0], model.slices[i_pop]].add_(model.Ja0[:, i_pop])

    for i_pop in range(model.N_POP):
        ff_input[:, model.N_STIM_ON[0]:, model.slices[i_pop]].add_(model.Ja0[:, i_pop])

    if model.TASK != 'None':
        for i ,_ in enumerate(model.N_STIM_ON):
            size = (model.N_BATCH, model.Na[0])
            if 'dual' in model.TASK:
                if 'rand' in model.TASK:
                    # random phase on the lr ring
                    theta = get_theta(model.PHI0[0], model.PHI0[2])
                    phase = torch.rand(model.size[0], dtype=model.dtype, device=model.device) * 360
                    model.phase = phase.unsqueeze(1).expand((phase.shape[0], model.size[-1]))

                    stimulus = Stimuli('odr', size, device=model.device)(model.I0[i],
                                                                        model.SIGMA0[i],
                                                                        model.phase,
                                                                        rnd_phase=0,
                                                                        theta_list=theta)
                    del theta

                    stimulus = stimulus.unsqueeze(1).expand((stimulus.shape[0],
                                                                1,
                                                                stimulus.shape[-1]))
                else:
                    stimulus = Stimuli(model.TASK, size)(model.I0[i], model.SIGMA0[i], model.PHI0[2*i+1])

            else:
                rnd_phase = 0
                if 'rand' in model.TASK:
                    rnd_phase = 1

                stimulus = Stimuli(model.TASK, size, device=model.device)(model.I0[i],
                                                                        model.SIGMA0[i],
                                                                        model.PHI0[:, i],
                                                                        rnd_phase=rnd_phase).unsqueeze(1)

            ff_input[:, model.N_STIM_ON[i]:model.N_STIM_OFF[i], model.slices[0]].add_(stimulus)

            del stimulus

    return ff_input * torch.sqrt(model.Ka[0]) * model.M0

def init_ff_input(model):
    if model.LIVE_FF_UPDATE:
        return init_ff_live(model)
    return init_ff_seq(model)
