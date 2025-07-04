import torch
import torch.nn as nn
import torch.nn.functional as F

class SignBCELoss(nn.Module):
      def __init__(self, alpha=1.0, thresh=2.0, imbalance=0):
            super(SignBCELoss, self).__init__()
            self.alpha = alpha
            self.thresh = thresh

            self.imbal = imbalance
            self.bce_with_logits = nn.BCEWithLogitsLoss()

      def imbal_func(target, imbalance):
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
            if self.imbal == -1:
                  # sign_overlap = torch.abs(torch.sign(2 * targets - 1)) * mean_readout
                  sign_overlap = torch.sign(targets) * mean_readout
                  self.imbal = 0
            else:
                  sign_overlap = torch.sign(2 * targets - 1) * mean_readout

            if self.imbal > 1.0:
                  sign_loss = F.relu(torch.sign(targets) * self.thresh - self.imb_func(targets, self.imbal) * sign_overlap)
            elif self.imbal == 0:
                  sign_loss = F.relu(self.imbal_func(targets, self.imbal) * self.thresh - sign_overlap)
            else:
                  sign_loss = F.relu(self.thresh - sign_overlap)

            combined_loss = (1-self.alpha) * bce_loss + self.alpha * sign_loss

            return combined_loss.mean()
