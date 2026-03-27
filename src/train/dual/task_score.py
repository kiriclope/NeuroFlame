import numpy as np
import torch
import torch.nn as nn


def calculate_mean_accuracy_and_sem(accuracies):
    accuracies = torch.as_tensor(accuracies, dtype=torch.float32).flatten()
    mean_accuracy = accuracies.mean().item()

    if accuracies.numel() < 2:
        return mean_accuracy, 0.0

    sem = accuracies.std(unbiased=True).item() / np.sqrt(accuracies.numel())
    return mean_accuracy, sem


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, readout, targets):
        prob = torch.sigmoid(readout)

        flip_mask = targets[:, 0] == 0
        while flip_mask.ndim < prob.ndim:
            flip_mask = flip_mask.unsqueeze(-1)

        return torch.where(flip_mask, 1.0 - prob, prob)


class DualScore(nn.Module):
    def __init__(self, cue_idx=None, rwd_idx=None, read_idx=None, device="cuda"):
        super().__init__()

        self.cue_idx = torch.as_tensor(
            [] if cue_idx is None else cue_idx,
            dtype=torch.long,
            device=device,
        )
        self.rwd_idx = torch.as_tensor(
            [] if rwd_idx is None else rwd_idx,
            dtype=torch.long,
            device=device,
        )
        self.read_idx = [-1] if read_idx is None else list(read_idx)
        self.score = Accuracy()

    def forward(self, readout, targets):
        if self.cue_idx.numel() == 0:
            return self.score(
                readout[:, self.rwd_idx, self.read_idx[0]],
                targets,
            )

        dpa_score = self.score(
            readout[:, self.rwd_idx, self.read_idx[0]],
            targets[:, 0, : self.rwd_idx.numel()],
        )
        drt_score = self.score(
            readout[:, self.cue_idx, self.read_idx[1]],
            targets[:, 2, : self.cue_idx.numel()],
        )
        return dpa_score, drt_score
