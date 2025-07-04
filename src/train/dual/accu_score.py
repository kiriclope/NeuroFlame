import torch

class Accuracy(nn.Module):
      def __init__(self, thresh=4.0):
            super(Accuracy, self).__init__()
            self.thresh = thresh

      def forward(self, readout, targets):
            mean_readout = readout.mean(dim=1)
            sign_loss = (mean_readout >= self.thresh)
            return 1.0 * (sign_loss == targets[:, 0])
