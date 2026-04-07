from __future__ import annotations

import numpy as np
import torch


def get_dual_indices(model):
    """
    Reproduce the dual-task evaluation windows from the notebook.
    Returns numpy index arrays.
    """
    steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)

    mask_rwd = (steps >= (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    rwd_idx = np.where(mask_rwd)[0]

    mask_cue = (
        (steps >= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY))
        & (steps <= (model.N_STIM_OFF[3].cpu().numpy() - model.N_STEADY))
    )
    cue_idx = np.where(mask_cue)[0]

    mask_gng = (
        (steps >= (model.N_STIM_OFF[1].cpu().numpy() - model.N_STEADY))
        & (steps <= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY))
    )
    gng_idx = np.where(mask_gng)[0]

    mask_stim = (
        (steps >= (model.N_STIM_ON[0].cpu().numpy() - model.N_STEADY))
        & (steps <= (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    )
    stim_idx = np.where(mask_stim)[0]

    mask_zero = ~mask_rwd & ~mask_cue & ~mask_stim
    zero_idx = np.where(mask_zero)[0]

    return {
        "stim_idx": stim_idx,
        "gng_idx": gng_idx,
        "cue_idx": cue_idx,
        "rwd_idx": rwd_idx,
        "zero_idx": zero_idx,
    }


def make_dual_test_data(model, batch_size=16, device=None):
    """
    Reproduce the 'Dual Task -> Testing -> Simulations' block.
    Returns:
        ff_input: [n_trials, T, N_in]
        labels:   [n_trials, 4, win]
        idx: dict of evaluation indices
    """
    if device is None:
        device = model.device if hasattr(model, "device") else "cpu"

    idx = get_dual_indices(model)

    model.N_BATCH = batch_size
    model.lr_eval_win = max(len(idx["rwd_idx"]), len(idx["cue_idx"]))

    ff_input = []
    labels = np.zeros((4, 12, model.N_BATCH, model.lr_eval_win), dtype=np.float32)

    l = 0
    for j in [0, 1, -1]:
        for i in [-1, 1]:
            for k in [-1, 1]:
                model.I0[0] = i  # sample
                model.I0[1] = j  # distractor
                model.I0[4] = k  # test

                if k == 1:
                    labels[3, l] = np.ones((model.N_BATCH, model.lr_eval_win), dtype=np.float32)

                if i == 1:
                    labels[1, l] = np.ones((model.N_BATCH, model.lr_eval_win), dtype=np.float32)

                if i == k:
                    labels[0, l] = np.ones((model.N_BATCH, model.lr_eval_win), dtype=np.float32)

                if j == 1:
                    model.I0[2] = 1
                    labels[2, l] = np.ones((model.N_BATCH, model.lr_eval_win), dtype=np.float32)
                elif j == -1:
                    model.I0[2] = 1
                    labels[2, l] = -np.ones((model.N_BATCH, model.lr_eval_win), dtype=np.float32)
                else:
                    model.I0[2] = 0

                l += 1
                with torch.no_grad():
                    ff_input.append(model.init_ff_input())

    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    labels = labels.reshape(4, -1, model.lr_eval_win).transpose(0, 1)  # [trials, 4, win]
    ff_input = torch.vstack(ff_input).to(device)

    return ff_input, labels, idx
