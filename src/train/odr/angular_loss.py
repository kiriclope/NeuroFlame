import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions


class CircularAngleLoss(nn.Module):
    def __init__(self, mode='angular', reduction='mean'):
        super().__init__()
        self.mode = mode
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred_angle, target_angle):
        if self.mode == 'polar':
            pred_sin, pred_cos = torch.sin(pred_angle), torch.cos(pred_angle)
            target_sin, target_cos = torch.sin(target_angle), torch.cos(target_angle)
            loss_sin = self.mse(pred_sin, target_sin)
            loss_cos = self.mse(pred_cos, target_cos)
            return (loss_sin + loss_cos) / 2

        elif self.mode == 'angular':
            error = 1 - torch.cos(pred_angle - target_angle)
            if self.reduction == 'mean':
                return error.mean()
            elif self.reduction == 'sum':
                return error.sum()
            else:
                return error
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")



class VonMisesNLLLoss(nn.Module):
    def __init__(self, kappa=4.0, reduction='none'):
        super().__init__()
        self.kappa = kappa
        self.reduction = reduction

    def forward(self, pred_angle, target_angle):
        # pred_angle and target_angle in radians, same shape
        vm = torch.distributions.VonMises(pred_angle, self.kappa)
        nll = -vm.log_prob(target_angle)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll  # (no reduction)


class AngularErrorLoss(nn.Module):
    def __init__(self, thresh=1.0, reg_tuning=0.1):
        super(AngularErrorLoss, self).__init__()

        self.loss = nn.MSELoss(reduction='none')
        # self.loss = nn.SmoothL1Loss(reduction='none')

        # self.polar_loss = VonMisesNLLLoss(reduction='none')
        self.polar_loss = CircularAngleLoss(reduction='none')

        self.thresh = thresh
        self.reg_tuning = reg_tuning

    def forward(self, readout, theta_batch):
        m0, m1, y_pred = decode_bump_torch(readout, axis=-1, device=readout.device)

        valid_mask = theta_batch != -999
        invalid_mask = ~valid_mask
        total_loss = 0

        # angular loss (Dcos, Dsin)
        loss_polar = self.polar_loss(theta_batch, y_pred) * valid_mask
        loss_angular = loss_polar.sum()
        total_loss += loss_angular

        # imposing tuning strength
        regularization = F.relu((self.thresh * m0 - m1)) * valid_mask
        # regularization = F.relu((1.0 - m1 / (self.thresh * m0 + 1e-6))) * valid_mask
        total_loss += self.reg_tuning * regularization.sum()

        # normalize over batch and time points
        total_loss /= valid_mask.sum()

        # imposing zero tuning in invalid mask
        loss_zero = self.loss(m1, 0.0 * m1) * invalid_mask
        total_loss += loss_zero.sum() / invalid_mask.sum()

        return total_loss
