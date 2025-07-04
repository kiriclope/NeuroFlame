import numpy as np
import torch
import torch.optim as optim

from time import perf_counter

from src.train.utils import convert_seconds
from src.train.dual.train_dpa import create_model
from src.train.split import split_data, cross_val_data
from src.train.dual.task_loss import DualLoss
from src.train.dual.optim import optimization


def load_naive_model(model, path, seed):
      model_state_dict = torch.load(path + '/dual_naive_%d.pth' % seed)
      model.load_state_dict(model_state_dict)

      model.J_STP.requires_grad = False
      model.low_rank.lr_kappa.requires_grad = False

      # for name, param in model.named_parameters():
      #       if param.requires_grad:
      #             print(name, param.shape)

      return model

def create_dual_masks(model):

    steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)

    mask_rwd = (steps >= (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    rwd_idx = np.where(mask_rwd)[0]
    # print('rwd', rwd_idx)

    mask_cue = (steps >= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_OFF[3].cpu().numpy() - model.N_STEADY))
    cue_idx = np.where(mask_cue)[0]
    print('cue', cue_idx)

    mask_GnG = (steps >= (model.N_STIM_OFF[1].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY))
    GnG_idx = np.where(mask_GnG)[0]
    # print('GnG', GnG_idx)

    mask_stim = (steps >= (model.N_STIM_ON[0].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    stim_idx = np.where(mask_stim)[0]
    # print('stim', stim_idx)

    mask_zero = ~mask_rwd & ~mask_cue & ~mask_stim
    zero_idx = np.where(mask_zero)[0]
    print('zero', zero_idx)

    return rwd_idx, cue_idx, stim_idx, zero_idx


def create_dual_input_labels(model):
    ff_input = []
    labels = np.zeros((3, 12, model.N_BATCH, model.lr_eval_win))

    l=0
    for i in [-1, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 1]:

                model.I0[0] = i # sample
                model.I0[1] = j # distractor
                model.I0[4] = k # test

                if i==1:
                    labels[1, l] = np.ones((model.N_BATCH, model.lr_eval_win))

                if i==k: # Pair Trials
                    labels[0, l] = np.ones((model.N_BATCH, model.lr_eval_win))

                if j==1: # Go
                    model.I0[2] = float(B0) # cue
                    model.I0[3] = float(C0) * model.IF_RL # rwd

                    labels[2, l] = np.ones((model.N_BATCH, model.lr_eval_win))

                elif j==-1: # NoGo
                    model.I0[2] = float(B0) # cue
                    model.I0[3] = 0.0 # rwd

                else: # DPA
                    model.I0[2] = 0 # cue
                    model.I0[3] = 0 # rwd

                l+=1

                ff_input.append(model.init_ff_input())

    labels = torch.tensor(labels, dtype=torch.float, device=DEVICE).reshape(3, -1, model.lr_eval_win).transpose(0, 1)
    ff_input = torch.vstack(ff_input)
    print('ff_input', ff_input.shape, 'labels', labels.shape)

    return ff_input, labels


def train_dual(REPO_ROOT, conf_name, seed, DEVICE):

    N_BATCH = 64
    batch_size = 16
    learning_rate = 0.1
    num_epochs = 15
    path = '../models/dual'

    model = create_model(REPO_ROOT, conf_name, seed, DEVICE)
    model = load_naive_model(model, path, seed)
    rwd_idx, cue_idx, stim_idx, zero_idx = create_dual_masks(model)

    model.N_BATCH = N_BATCH
    model.lr_eval_win = np.max( (rwd_idx.shape[0], cue_idx.shape[0]))

    ff_input, labels = create_dual_input_labels(model)

    train_loader, val_loader = split_data(ff_input, labels, train_perc=0.8, batch_size=batch_size)
    criterion = DualLoss(alpha=1.0, thresh=4.0, stim_idx=stim_idx, cue_idx=cue_idx, rwd_idx=rwd_idx, zero_idx=zero_idx, imbalance=[1.0, 0.0], read_idx=[1, 0, 1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training Dual')
    num_epochs = 15
    start = perf_counter()

    loss, val_loss = optimization(model, train_loader, val_loader, criterion, optimizer, num_epochs, zero_grad=None)
    end = perf_counter()
    print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

    torch.save(model.state_dict(), '../models/dual/dual_train_%d.pth' % seed)
