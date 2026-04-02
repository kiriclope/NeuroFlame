import numpy as np
import torch
import torch.optim as optim

from time import perf_counter

from src.train.utils import convert_seconds
from src.network import Network
from src.utils import clear_cache
from src.train.split import split_data, cross_val_data
from src.train.dual.task_loss import DualLoss
from src.train.dual.optim import optimization


def create_model(REPO_ROOT, conf_name, seed, DEVICE, **kwargs):

    model = Network(conf_name, REPO_ROOT, VERBOSE=0, DEVICE=DEVICE, SEED=seed, N_BATCH=1, **kwargs)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.J_STP.requires_grad = True

    if model.LR_READOUT:
        for param in model.low_rank.linear.parameters():
            param.requires_grad = False
        model.low_rank.linear.bias.requires_grad = False

    if model.LR_KAPPA:
        model.low_rank.lr_kappa.requires_grad = True

    return model


def create_dpa_masks(model):
    steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)
    mask = (steps >= (model.N_STIM_ON[4].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_OFF[-1].cpu().numpy() - model.N_STEADY + 1))
    rwd_idx = np.where(mask)[0]
    # print('rwd', rwd_idx)

    # mask for A/B memory from sample to test
    stim_mask = (steps >= (model.N_STIM_ON[0].cpu().numpy() - model.N_STEADY)) & (steps < (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    stim_idx = np.where(stim_mask)[0]
    # print('stim', stim_idx)

    model.lr_eval_win = np.max((rwd_idx.shape[0], stim_idx.shape[0]))

    mask_zero = ~mask  # & ~stim_mask
    zero_idx = np.where(mask_zero)[0]
    # print('zero', zero_idx)

    return rwd_idx, stim_idx, zero_idx


def create_dpa_input_labels(model):

    ff_input = []
    labels = np.zeros((2, 4, model.N_BATCH, model.lr_eval_win))

    l=0
    for i in [-1, 1]:
        for k in [-1, 1]:

            model.I0[0] = i # sample
            model.I0[4] = k # test

            if i == 1:
                    labels[1, l] = np.ones((model.N_BATCH, model.lr_eval_win))

            if i==k: # Pair Trials
                labels[0, l] = np.ones((model.N_BATCH, model.lr_eval_win))

            l+=1

            ff_input.append(model.init_ff_input())

    labels = torch.tensor(labels, dtype=torch.float, device=model.device).reshape(2, -1, model.lr_eval_win).transpose(0, 1)

    ff_input = torch.vstack(ff_input)
    print('ff_input', ff_input.shape, 'labels', labels.shape)

    return ff_input, labels


def train_dpa(REPO_ROOT, conf_name, seed, DEVICE):

    N_BATCH = 256
    batch_size = 16
    learning_rate = 0.1

    model = create_model(REPO_ROOT, conf_name, seed, DEVICE)
    path = model.SAVE_PATH
    print(path)

    rwd_idx, stim_idx, zero_idx = create_dpa_masks(model)

    model.N_BATCH = N_BATCH
    model.lr_eval_win = np.max((rwd_idx.shape[0], stim_idx.shape[0]))

    ff_input, labels = create_dpa_input_labels(model)
    splits = [split_data(ff_input, labels, train_perc=0.8, batch_size=batch_size)]
    # # splits = cross_val_data(ff_input, labels, n_splits=5, batch_size=batch_size)

    del ff_input, labels

    criterion = DualLoss(alpha=0.0, thresh=2.0, rwd_idx=rwd_idx, stim_idx=stim_idx, zero_idx=zero_idx,
                         class_bal=[1.0, 1.0], read_idx=[1, 0], DEVICE=DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training DPA')

    start = perf_counter()
    for train_loader, val_loader in splits:
        optimization(model, train_loader, val_loader, criterion, optimizer, zero_grad=None)
    end = perf_counter()

    print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

    torch.save(model.state_dict(), path + '/dpa_%d.pth' % seed)

    del model
    clear_cache()
