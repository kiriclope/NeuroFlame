import numpy as np
import torch
import torch.optim as optim

from time import perf_counter

from src.train.utils import convert_seconds
from src.train.dual.train_dpa import create_model
from src.train.split import split_data, cross_val_data
from src.train.dual.task_loss import DualLoss
from src.train.dual.optim import optimization


def load_dpa_model(model, path, seed):
      model_state_dict = torch.load(path + '/dpa_%d.pth' % seed)
      model.load_state_dict(model_state_dict)

      model.J_STP.requires_grad = False
      model.low_rank.lr_kappa.requires_grad = False

      # for name, param in model.named_parameters():
      #       if param.requires_grad:
      #             print(name, param.shape)

      return model

def create_gng_masks(model):
      steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)

      # mask for lick/nolick  from cue to test
      rwd_mask = (steps >= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY)) & (steps < (model.N_STIM_ON[4].cpu().numpy() - model.N_STEADY))
      rwd_idx = np.where(rwd_mask)[0]
      # print('rwd', rwd_idx)

      # mask for Go/NoGo memory from dist to cue
      stim_mask = (steps >= (model.N_STIM_ON[1].cpu().numpy() - model.N_STEADY)) & (steps < (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY))
      stim_idx = np.where(stim_mask)[0]
      # stim_idx = []
      # print('stim', stim_idx)

      mask_zero = (steps < (model.N_STIM_ON[1].cpu().numpy() - model.N_STEADY))
      zero_idx = np.where(mask_zero)[0]
      # print('zero', zero_idx)

      return rwd_idx, stim_idx, zero_idx

def create_gng_input_labels(model):
      for i in range(5):
            model.I0[i] = 0.0
      model.I0[2] = 1.0 # cue

      ff_input = []
      for i in [-1.0, 1.0]:
            model.I0[1] = i
            ff_input.append(model.init_ff_input())

      ff_input = torch.vstack(ff_input)

      labels_Go = torch.ones((model.N_BATCH, model.lr_eval_win))
      labels_NoGo = torch.zeros((model.N_BATCH, model.lr_eval_win))
      labels = torch.cat((labels_NoGo, labels_Go))

      print('ff_input', ff_input.shape, 'labels', labels.shape)

      labels = labels.repeat((2, 1, 1))
      labels = torch.transpose(labels, 0, 1)
      print('labels', labels.shape)

      return ff_input, labels

def train_gng(REPO_ROOT, conf_name, seed, DEVICE):

    N_BATCH = 512
    batch_size = 16
    learning_rate = 0.1
    num_epochs = 15
    path = '../models/dual'

    model = create_model(REPO_ROOT, conf_name, seed, DEVICE)
    model = load_dpa_model(model, path, seed)
    rwd_idx, stim_idx, zero_idx = create_gng_masks(model)

    model.N_BATCH = N_BATCH
    model.lr_eval_win = np.max((rwd_idx.shape[0], stim_idx.shape[0]))

    ff_input, labels = create_gng_input_labels(model)

    train_loader, val_loader = split_data(ff_input, labels, train_perc=0.8, batch_size=batch_size)
    criterion = DualLoss(alpha=1.0, thresh=4.0, rwd_idx=rwd_idx, zero_idx=zero_idx, stim_idx=stim_idx, imbalance=[0.0, 1.0], read_idx=[1, 1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training DRT')

    start = perf_counter()
    loss, val_loss = optimization(model, train_loader, val_loader, criterion, optimizer, num_epochs, zero_grad=0)
    end = perf_counter()
    print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

    torch.save(model.state_dict(), path + '/dual_naive_%d.pth' % seed)
