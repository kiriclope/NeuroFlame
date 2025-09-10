import torch
import torch.nn as nn

from src.train.dual.sign_loss import SignBCELoss

class DualLoss(nn.Module):
      def __init__(self, alpha=1.0, thresh=5.0, stim_idx=[], gng_idx=[], cue_idx=[], rwd_idx=-1, zero_idx=[], read_idx=[-1], class_bal=[0], DEVICE='cuda'):
            super(DualLoss, self).__init__()
            self.alpha = alpha
            self.thresh = thresh
            self.class_bal = class_bal

            # BL idx
            self.zero_idx = zero_idx
            # Sample idx
            self.stim_idx = torch.tensor(stim_idx, dtype=torch.int, device=DEVICE)
            # Go NoGo
            self.gng_idx= torch.tensor(gng_idx, dtype=torch.int, device=DEVICE)
            # rwd idx for DRT
            self.cue_idx = torch.tensor(cue_idx, dtype=torch.int, device=DEVICE)
            # rwd idx for DPA
            self.rwd_idx = torch.tensor(rwd_idx, dtype=torch.int, device=DEVICE)

            # readout idx
            self.read_idx = read_idx

            self.loss = SignBCELoss(self.alpha, self.thresh)
            self.l1loss = nn.SmoothL1Loss()


      def forward(self, readout, targets):

            zeros = torch.zeros_like(readout[:, self.zero_idx, 0])
            # custom zeros for readout
            loss = self.l1loss(readout[:, self.zero_idx, self.read_idx[0]], zeros)
            # zero memory only before stim
            if len(self.read_idx)>1:
                  loss += self.l1loss(readout[:, :self.stim_idx[0]-1, self.read_idx[1]], zeros[:, :self.stim_idx[0]-1])

            is_stim = (self.stim_idx.numel() != 0)
            is_gng = (self.gng_idx.numel() != 0)
            is_cue = (self.cue_idx.numel() != 0)
            is_rwd = (self.rwd_idx.numel() != 0)

            if is_cue:
                  self.loss.class_bal = self.class_bal[1]
                  loss += self.loss(readout[:, self.cue_idx, self.read_idx[3]], targets[:, 2, :self.cue_idx.shape[0]])

            if is_gng:
                  self.loss.class_bal = 1
                  loss += self.loss(readout[:,  self.gng_idx, self.read_idx[2]], targets[:, 2, :self.gng_idx.shape[0]])

            if is_stim:
                  self.loss.class_bal = 1
                  loss += self.loss(readout[:,  self.stim_idx, self.read_idx[1]], targets[:, 1, :self.stim_idx.shape[0]])

            if is_rwd:
                  self.loss.class_bal = self.class_bal[0]
                  loss += self.loss(readout[:,  self.rwd_idx, self.read_idx[0]], targets[:, 0, :self.rwd_idx.shape[0]])

            return loss
