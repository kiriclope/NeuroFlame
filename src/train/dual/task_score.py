import torch
import torch.nn as nn
import numpy as np

def calculate_mean_accuracy_and_sem(accuracies):
    mean_accuracy = accuracies.mean()
    std_dev = accuracies.std(unbiased=True).item()
    sem = std_dev / np.sqrt(len(accuracies))
    return mean_accuracy, sem


class Accuracy(nn.Module):
    def __init__(self, thresh=4.0):
        super(Accuracy, self).__init__()
        self.thresh = thresh


    def imbal_func(self, target, imbalance=0):
        output = torch.zeros_like(target)

        output[target == 0] = imbalance
        output[target == 1] = 1

        return output


    def forward(self, readout, targets, imbalance=0):
        # mean_readout = readout.mean(dim=1)
        # sign_overlap = torch.sign(2 * targets[:, 0] - 1) * mean_readout
        # return 1.0 * (sign_overlap >= self.thresh)

        # sign_loss = (mean_readout >= self.thresh)
        # return 1.0 * (sign_loss == targets[:, 0])

        # sign_readout = torch.sign(2 * targets[:, 0].unsqueeze(-1) - 1) * readout
        # accuracy = 1.0 * (sign_readout >= self.imbal_func(targets[:,0], imbalance).unsqueeze(-1) * self.thresh).any(dim=1)

        prob = torch.sigmoid(readout[..., -1])
        idx = torch.where(targets[:, 0]==0)
        prob[idx] = 1 - prob[idx]
        # print(prob.shape)
        # predicted = (prob >= 0.5).float()
        accuracy = prob

        # target_vals = self.imbal_func(targets[:, 0], imbalance).unsqueeze(-1)  # shape: (batch, 1)
        # target_vals = targets[:, 0].unsqueeze(-1)  # shape: (batch, 1)
        # accuracy = (predicted == target_vals).float()

        return accuracy


class DualScore(nn.Module):
    def __init__(self, thresh=2.0, cue_idx=[], rwd_idx=-1, read_idx=[-1], DEVICE='cuda'):
        super().__init__()

        self.thresh = thresh
        # rwd idx for DRT
        self.cue_idx = torch.tensor(cue_idx, dtype=torch.int, device=DEVICE)
        # rwd idx for DPA
        self.rwd_idx = torch.tensor(rwd_idx, dtype=torch.int, device=DEVICE)

        # readout idx
        self.read_idx = read_idx
        self.score = Accuracy(thresh=self.thresh)

    def forward(self, readout, targets):
        targ = targets.clone()
        targ[targ==-1] = 0
        is_empty = (self.cue_idx.numel() == 0)

        if is_empty:
            DPA_score = self.score(readout[:, self.rwd_idx, self.read_idx[0]], targ)
            return DPA_score

        DPA_score = self.score(readout[:, self.rwd_idx, self.read_idx[0]], targ[:, 0, :self.rwd_idx.shape[0]], imbalance=0)
        DRT_score = self.score(readout[:, self.cue_idx, self.read_idx[1]], targ[:, -1, :self.cue_idx.shape[0]], imbalance=0)

        return DPA_score, DRT_score
