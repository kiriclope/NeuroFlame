import torch
import torch.nn as nn
import torch.nn.functional as F

class SignBCELoss(nn.Module):
      def __init__(self, alpha=1.0, thresh=2.0, imbalance=0):
            super(SignBCELoss, self).__init__()
            self.alpha = alpha
            self.thresh = thresh

            self.imbalance = imbalance
            self.bce_with_logits = nn.BCEWithLogitsLoss()

      def imbal_func(self, target, imbalance):
          output = torch.zeros_like(target)

          output[target == 0] = imbalance
          output[target == 1] = 1

          return output

      def forward(self, readout, targets):
            if self.alpha != 1.0:
                  bce_loss = self.bce_with_logits(readout, targets)
            else:
                  bce_loss = 0.0

            # average readout over bins
            mean_readout = readout.mean(dim=1).unsqueeze(-1)

            # only penalizing not licking when pair
            if self.imbalance == -1:
                  # sign_overlap = torch.abs(torch.sign(2 * targets - 1)) * mean_readout
                  sign_overlap = torch.sign(targets) * mean_readout
                  self.imbalance = 0
            else:
                  sign_overlap = torch.sign(2 * targets - 1) * mean_readout

            if self.imbalance > 1.0:
                  sign_loss = F.relu(torch.sign(targets) * self.thresh - self.imbal_func(targets, self.imbalance) * sign_overlap)
            elif self.imbalance == 0:
                  sign_loss = F.relu(self.imbal_func(targets, self.imbalance) * self.thresh - sign_overlap)
            else:
                  sign_loss = F.relu(self.thresh - sign_overlap)

            combined_loss = (1-self.alpha) * bce_loss + self.alpha * sign_loss

            return combined_loss.mean()

import torch
import torch.nn as nn

# from src.train.dual.signloss import SignBCELoss

class DualLoss(nn.Module):
      def __init__(self, alpha=1.0, thresh=2.0, stim_idx=[], cue_idx=[], rwd_idx=-1, zero_idx=[], read_idx=[-1], imbalance=0, DEVICE='cuda'):
            super(DualLoss, self).__init__()
            self.alpha = alpha
            self.thresh = thresh
            self.imbalance = imbalance

            # BL idx
            self.zero_idx = zero_idx
            # Sample idx
            self.stim_idx = torch.tensor(stim_idx, dtype=torch.int, device=DEVICE)
            # rwd idx for DRT
            self.cue_idx = torch.tensor(cue_idx, dtype=torch.int, device=DEVICE)
            # rwd idx for DPA
            self.rwd_idx = torch.tensor(rwd_idx, dtype=torch.int, device=DEVICE)

            # readout idx
            self.read_idx = read_idx

            self.loss = SignBCELoss(self.alpha, self.thresh, self.imbalance)
            # self.l1loss = nn.SmoothL1Loss()
            self.l1loss = nn.MSELoss()

      def forward(self, readout, targets):

            zeros = torch.zeros_like(readout[:, self.zero_idx, 0])
            # custom zeros for readout
            loss = self.l1loss(readout[:, self.zero_idx, self.read_idx[0]], zeros)
            # zero memory only before stim
            if len(self.read_idx)>1:
                  loss += self.l1loss(readout[:, :self.stim_idx[0]-1, self.read_idx[1]], zeros[:, :self.stim_idx[0]-1])

            is_stim = (self.stim_idx.numel() != 0)
            is_cue = (self.cue_idx.numel() != 0)
            is_rwd = (self.rwd_idx.numel() != 0)

            if is_cue:
                  self.loss.imbalance = self.imbalance[1]
                  loss += self.loss(readout[:, self.cue_idx, self.read_idx[2]], targets[:, 2, :self.cue_idx.shape[0]])
            if is_stim:
                  self.loss.imbalance = 1
                  loss += self.loss(readout[:,  self.stim_idx, self.read_idx[1]], targets[:, 1, :self.stim_idx.shape[0]])
            if is_rwd:
                  self.loss.imbalance = self.imbalance[0]
                  loss += self.loss(readout[:,  self.rwd_idx, self.read_idx[0]], targets[:, 0, :self.rwd_idx.shape[0]])

            return loss
