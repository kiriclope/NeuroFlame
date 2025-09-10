import gc
import numpy as np
import torch
import torch.optim as optim

from time import perf_counter

from src.train.utils import convert_seconds
from src.train.dual.train_dpa import create_model
from src.train.split import split_data, cross_val_data
from src.train.dual.task_loss import DualLoss
from src.train.dual.task_score import DualScore, calculate_mean_accuracy_and_sem
from src.train.dual.optim import optimization
from src.train.dual.covariance import compute_cov
from src.utils import clear_cache


def del_tensor(tensor):
    DEVICE = tensor.device
    del tensor
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.device(DEVICE)
    torch.cuda.synchronize()
    torch.cuda.reset_accumulated_memory_stats(DEVICE)


def load_model(model, path, seed, state='naive'):
    if state=='dpa':
        model_state_dict = torch.load(path + '/%s_%d.pth' % (state, seed))
    else:
        model_state_dict = torch.load(path + '/dual_%s_%d.pth' % (state, seed))

    model.load_state_dict(model_state_dict)

    model.J_STP.requires_grad = False

    # for name, param in model.named_parameters():
    #       if param.requires_grad:
    #             print(name, param.shape)

    return model

def create_dual_masks(model):

    steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)

    mask_rwd = (steps >= (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    # & (steps <= (model.N_STIM_OFF[-1].cpu().numpy() - model.N_STEADY))
    rwd_idx = np.where(mask_rwd)[0]
    # print('rwd', rwd_idx)

    mask_cue = (steps >= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_OFF[3].cpu().numpy() - model.N_STEADY))
    cue_idx = np.where(mask_cue)[0]
    # print('cue', cue_idx)

    mask_GnG = (steps >= (model.N_STIM_ON[1].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY))
    gng_idx = np.where(mask_GnG)[0]
    # print('GnG', GnG_idx)

    mask_stim = (steps>=(model.N_STIM_ON[0].cpu().numpy() - model.N_STEADY)) & (steps<=(model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
    stim_idx = np.where(mask_stim)[0]
    # print('stim', stim_idx)

    mask_zero = ~mask_rwd & ~mask_cue & ~mask_stim
    zero_idx = np.where(mask_zero)[0]
    # print('zero', zero_idx)

    return rwd_idx, cue_idx, gng_idx, stim_idx, zero_idx


def create_dual_input_labels(model):
    ff_input = []
    labels = np.zeros((3, 12, model.N_BATCH, model.lr_eval_win))

    l=0

    for j in [0, 1, -1]:
        for i in [-1, 1]:
            for k in [-1, 1]:

                model.I0[0] = i # sample
                model.I0[1] = j # distractor
                model.I0[4] = k # test

                if i==1: # sample A trials
                    labels[1, l] = np.ones((model.N_BATCH, model.lr_eval_win))

                if i==k: # Pair trials
                    labels[0, l] = np.ones((model.N_BATCH, model.lr_eval_win))

                if j==1: # Go trials
                    model.I0[2] = 1.0 # cue
                    model.I0[3] = 0.0 # GnG rwd

                    labels[2, l] = np.ones((model.N_BATCH, model.lr_eval_win))

                elif j==-1: # NoGo trials
                    model.I0[2] = 1.0 # cue
                    model.I0[3] = 0.0 # GnG rwd

                    labels[2, l] = -np.ones((model.N_BATCH, model.lr_eval_win))

                else: # DPA trials
                    model.I0[2] = 0 # cue
                    model.I0[3] = 0 # rwd

                l+=1

                ff_input.append(model.init_ff_input())

    labels = torch.tensor(labels, dtype=torch.float, device=model.device).reshape(3, -1, model.lr_eval_win).transpose(0, 1)
    ff_input = torch.vstack(ff_input)
    # print('ff_input', ff_input.shape, 'labels', labels.shape)

    return ff_input, labels


def get_accuracy(readout, y_labels, criterion):

    dpa_perf, drt_perf = criterion(readout, y_labels)

    dpa_mean = []
    dpa_sem = []
    for i in [0, 1, -1]:
        y = torch.where(y_labels[:, 2, 0]==i)
        mean_, sem_ = calculate_mean_accuracy_and_sem(dpa_perf[y])
        dpa_mean.append(mean_)
        dpa_sem.append(sem_)

    dpa_mean = torch.stack(dpa_mean).cpu().detach().numpy()
    dpa_sem = np.stack(dpa_sem)

    drt_mean, drt_sem = calculate_mean_accuracy_and_sem(drt_perf)
    print('Dual accuracy:', dpa_mean, 'GoNoGo:', drt_mean.item())

    acc_mean = np.concatenate((dpa_mean, [drt_mean.cpu().detach().numpy()]))
    acc_sem = np.concatenate((dpa_sem, [drt_sem]))

    return acc_mean, acc_sem


def train_dual(REPO_ROOT, conf_name, seed, DEVICE):

    N_BATCH = 64
    batch_size = 16
    learning_rate = 0.1

    model = create_model(REPO_ROOT, conf_name, seed, DEVICE)
    path = model.SAVE_PATH

    model = load_model(model, path, seed)
    rwd_idx, cue_idx, gng_idx, stim_idx, zero_idx = create_dual_masks(model)

    model.N_BATCH = N_BATCH
    model.lr_eval_win = np.max( (rwd_idx.shape[0], cue_idx.shape[0], stim_idx.shape[0], gng_idx.shape[0]))

    ff_input, labels = create_dual_input_labels(model)
    train_loader, val_loader = split_data(ff_input, labels, train_perc=0.8, batch_size=batch_size)
    del ff_input, labels

    criterion = DualLoss(alpha=0.0, thresh=2.0, stim_idx=stim_idx, gng_idx=gng_idx, cue_idx=cue_idx, rwd_idx=rwd_idx, zero_idx=zero_idx, class_bal=[1.0, 0.0, 1.0, 1.0], read_idx=[1, 0, 1, 1])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training Dual')

    start = perf_counter()
    optimization(model, train_loader, val_loader, criterion, optimizer, zero_grad=None)
    end = perf_counter()

    print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

    torch.save(model.state_dict(), path + '/dual_train_%d.pth' % seed)

    del model
    clear_cache()


def test_dual(REPO_ROOT, conf_name, seed, state, thresh, DEVICE, IF_OPTO=0):

      N_BATCH = 64

      model = create_model(REPO_ROOT, conf_name, seed, DEVICE)
      path = model.SAVE_PATH
      model = load_model(model, path, seed, state)
      rwd_idx, cue_idx, gng_idx, stim_idx, zero_idx = create_dual_masks(model)

      # need that for computing loss on test offset (can't add to func because messes training)
      steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)
      mask_rwd = (steps >= (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY)) # & (steps <= (model.N_STIM_OFF[-1].cpu().numpy() - model.N_STEADY))
      rwd_idx = np.where(mask_rwd)[0]

      mask_cue = (steps >= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STIM_OFF[2].cpu().numpy() - model.N_STEADY))
      cue_idx = np.where(mask_cue)[0]

      model.N_BATCH = N_BATCH
      model.lr_eval_win = np.max((rwd_idx.shape[0], cue_idx.shape[0]))

      if IF_OPTO:
        model.IF_OPTO = 1

      ff_input, y_labels = create_dual_input_labels(model)
      # print('ff_input', ff_input.shape)

      print('Testing Dual')
      start = perf_counter()
      model.forward(ff_input=ff_input)
      readout = model.readout
      end = perf_counter()
      print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

      del_tensor(ff_input)
      cov_mn = compute_cov(model, order=0)
      cov_sample = compute_cov(model, order=1)
      cov_test = compute_cov(model, order=2)
      cov_go = compute_cov(model, order=3)
      cov = np.stack((cov_mn, cov_sample, cov_test, cov_go))

      del_tensor(model)

      criterion = DualScore(thresh=thresh, cue_idx=cue_idx, rwd_idx=rwd_idx, read_idx=[1, 1], DEVICE=DEVICE)
      accuracy = get_accuracy(readout, y_labels, criterion)

      y_labels = y_labels[..., 0].T.cpu().numpy()
      readout = readout.cpu().detach().numpy()

      return readout, y_labels, cov, accuracy
