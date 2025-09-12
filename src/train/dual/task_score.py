import torch
import torch.nn as nn
import numpy as np

def calculate_mean_accuracy_and_sem(accuracies):
    mean_accuracy = accuracies.mean()
    std_dev = accuracies.std(unbiased=True).item()
    sem = std_dev / np.sqrt(len(accuracies))
    return mean_accuracy, sem


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, readout, targets, class_bal=1):

        prob = torch.sigmoid(readout)
        idx = torch.where(targets[:, 0]==0)
        prob[idx] = (1 - prob[idx])

        # if class_bal==0:
        #     idx2 = torch.where(prob[idx]>=0.5)
        #     prob[idx2] = 1.0

        # accuracy = (prob >= 0.5).float()

        return prob


class DualScore(nn.Module):
    def __init__(self, cue_idx=[], rwd_idx=-1, read_idx=[-1], DEVICE='cuda'):
        super().__init__()

        # rwd idx for DRT
        self.cue_idx = torch.tensor(cue_idx, dtype=torch.int, device=DEVICE)
        # rwd idx for DPA
        self.rwd_idx = torch.tensor(rwd_idx, dtype=torch.int, device=DEVICE)

        # readout idx
        self.read_idx = read_idx

        self.score = Accuracy()


    def forward(self, readout, targets):
        is_empty = (self.cue_idx.numel() == 0)

        if is_empty:
            DPA_score = self.score(readout[:, self.rwd_idx, self.read_idx[0]], targets)
            return DPA_score

        DPA_score = self.score(readout[:, self.rwd_idx, self.read_idx[0]], targets[:, 0, :self.rwd_idx.shape[0]])
        DRT_score = self.score(readout[:, self.cue_idx, self.read_idx[1]], targets[:, -1, :self.cue_idx.shape[0]], class_bal=0)

        return DPA_score, DRT_score
