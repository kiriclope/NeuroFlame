import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def split_data(X, Y, train_perc=0.8, batch_size=32):
    if Y.ndim == 3:
        stratify = Y[:, 0, 0].detach().cpu().numpy()
    else:
        stratify = Y[:, 0].detach().cpu().numpy()

    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        train_size=train_perc,
        stratify=stratify,
        shuffle=True,
    )

    print("train X:", tuple(X_train.shape), "val X:", tuple(X_val.shape))
    print("train Y:", tuple(Y_train.shape), "val Y:", tuple(Y_val.shape))

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def get_steps(model):
    return np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)


def get_timing_arrays(model):
    return (
        model.N_STIM_ON.detach().cpu().numpy(),
        model.N_STIM_OFF.detach().cpu().numpy(),
        model.N_STEADY,
    )


def to_idx(mask):
    return np.where(mask)[0]


def set_eval_window(model, n_batch, *idx_arrays):
    valid_arrays = [arr for arr in idx_arrays if arr is not None]
    if not valid_arrays:
        raise ValueError("At least one index array is required to set eval window")

    model.N_BATCH = n_batch
    model.lr_eval_win = max(len(arr) for arr in valid_arrays)


def finalize_labels(labels, device, n_channels, eval_win):
    return (
        torch.tensor(labels, dtype=torch.float32, device="cpu")
        .reshape(n_channels, -1, eval_win)
        .transpose(0, 1)
    )


def finalize_idx(**kwargs):
    return {key: value.tolist() for key, value in kwargs.items()}


def build_ff_inputs(model, assignments):
    ff_input = []
    for assignment in assignments:
        for idx, value in assignment.items():
            model.I0[idx] = value
        ff_input.append(model.init_ff_input().cpu())
    return torch.vstack(ff_input)


def make_dpa_dataset(model, device, n_batch, rank):
    steps = get_steps(model)
    n_stim_on, _, steady = get_timing_arrays(model)

    rwd_idx = to_idx(steps >= (n_stim_on[-1] - steady))
    test_idx = rwd_idx if rank == 3 else np.array([], dtype=int)
    stim_idx = to_idx((steps >= (n_stim_on[0] - steady)) & (steps < (n_stim_on[-1] - steady)))
    zero_idx = to_idx(~(steps >= (n_stim_on[-1] - steady)))

    set_eval_window(model, n_batch, rwd_idx, stim_idx)

    labels = np.zeros((3, 4, model.N_BATCH, model.lr_eval_win), dtype=np.float32)
    assignments = []

    sample_idx = 0
    for i in (-1, 1):
        for k in (-1, 1):
            assignments.append({0: i, 4: k})

            if k == 1:
                labels[2, sample_idx] = 1.0
            if i == 1:
                labels[1, sample_idx] = 1.0
            if i == k:
                labels[0, sample_idx] = 1.0

            sample_idx += 1

    return (
        build_ff_inputs(model, assignments),
        finalize_labels(labels, device, n_channels=3, eval_win=model.lr_eval_win),
        finalize_idx(
            rwd_idx=rwd_idx,
            stim_idx=stim_idx,
            test_idx=test_idx,
            zero_idx=zero_idx,
        ),
    )


def make_gonogo_dataset(model, device, n_batch):
    steps = get_steps(model)
    n_stim_on, _, steady = get_timing_arrays(model)

    rwd_idx = to_idx((steps >= (n_stim_on[2] - steady)) & (steps < (n_stim_on[4] - steady)))
    stim_idx = to_idx((steps >= (n_stim_on[1] - steady)) & (steps < (n_stim_on[2] - steady)))
    zero_idx = to_idx(steps < (n_stim_on[1] - steady))

    set_eval_window(model, n_batch, rwd_idx, stim_idx)

    assignments = [
        {0: 0, 1: 1, 2: 1, 3: 0, 4: 0},
        {0: 0, 1: -1, 2: 1, 3: 0, 4: 0},
    ]
    ff_input = build_ff_inputs(model, assignments)

    labels_go = torch.ones((model.N_BATCH, model.lr_eval_win), device=device)
    labels_nogo = torch.zeros((model.N_BATCH, model.lr_eval_win), device=device)
    labels = torch.cat((labels_go, labels_nogo), dim=0).repeat((2, 1, 1)).transpose(0, 1)

    return ff_input, labels, finalize_idx(
        rwd_idx=rwd_idx,
        stim_idx=stim_idx,
        zero_idx=zero_idx,
    )


def make_dual_dataset(model, device, n_batch, rank):
    steps = get_steps(model)
    n_stim_on, n_stim_off, steady = get_timing_arrays(model)

    mask_rwd = steps >= (n_stim_on[-1] - steady)
    mask_cue = (steps >= (n_stim_on[2] - steady)) & (steps <= (n_stim_off[3] - steady))
    mask_gng = (steps >= (n_stim_on[1] - steady)) & (steps <= (n_stim_on[2] - steady))
    mask_stim = (steps >= (n_stim_on[0] - steady)) & (steps <= (n_stim_on[-1] - steady))
    mask_zero = ~mask_rwd & ~mask_cue & ~mask_stim

    rwd_idx = to_idx(mask_rwd)
    test_idx = rwd_idx if rank == 3 else np.array([], dtype=int)
    cue_idx = to_idx(mask_cue)
    gng_idx = to_idx(mask_gng)
    stim_idx = to_idx(mask_stim)
    zero_idx = to_idx(mask_zero)

    set_eval_window(model, n_batch, rwd_idx, cue_idx, stim_idx, gng_idx)

    labels = np.zeros((4, 12, model.N_BATCH, model.lr_eval_win), dtype=np.float32)
    assignments = []

    sample_idx = 0
    for i in (-1, 1):
        for j in (-1, 0, 1):
            for k in (-1, 1):
                assignment = {0: i, 1: j, 4: k}

                if j == 1:
                    assignment[2] = 1.0
                    assignment[3] = 0.0
                    labels[2, sample_idx] = 1.0
                elif j == -1:
                    assignment[2] = 1.0
                    assignment[3] = 0.0
                else:
                    assignment[2] = 0.0
                    assignment[3] = 0.0

                if k == 1:
                    labels[3, sample_idx] = 1.0
                if i == 1:
                    labels[1, sample_idx] = 1.0
                if i == k:
                    labels[0, sample_idx] = 1.0

                assignments.append(assignment)
                sample_idx += 1

    return (
        build_ff_inputs(model, assignments),
        finalize_labels(labels, device, n_channels=4, eval_win=model.lr_eval_win),
        finalize_idx(
            rwd_idx=rwd_idx,
            test_idx=test_idx,
            cue_idx=cue_idx,
            gng_idx=gng_idx,
            stim_idx=stim_idx,
            zero_idx=zero_idx,
        ),
    )
