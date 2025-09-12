import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_mean(tensor):
    """Returns mean or zero if tensor is empty."""
    if tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)

    return tensor.mean()

class BCEOneClassLoss(nn.Module):
    # Your original BCEOneClassLoss goes here
    def __init__(self, g_neutral=0.05):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.g_neutral = g_neutral

    def forward(self, logits, targets, class_bal=1):
        bce = self.criterion(logits, targets.float())

        if class_bal == 0:
            mask1 = (targets == 1)
            pos_loss = safe_mean(bce[mask1])
            mask0 = (targets == 0)
            # Encourage proba = 0.5 for class 0 (logit=0)
            neutral_loss = safe_mean(self.criterion(logits[mask0], torch.full_like(logits[mask0], 0.5)))
            return pos_loss + self.g_neutral * neutral_loss

        return safe_mean(bce)


class SignBCELoss(nn.Module):
    def __init__(self, alpha=0.5, thresh=1.0, class_bal=0, g_neutral=0.1):
        super().__init__()
        self.alpha = alpha
        self.thresh = thresh
        self.class_bal = class_bal
        self.bce_with_logits = BCEOneClassLoss()
        self.g_neutral = g_neutral

    def forward(self, readout, targets):
        # BCE loss (can be 0 if alpha==1)
        bce_loss = 0.0
        if self.alpha != 1.0:
            bce_loss = self.bce_with_logits(readout, targets, self.class_bal)

        sign_overlap = torch.sign(2 * targets - 1) * readout
        sign_loss = torch.zeros_like(sign_overlap)

        if self.alpha != 0.0:
            if self.class_bal == 0:
                # Penalize class 0 (targets==0) with |overlap|
                mask0 = (targets == 0)
                if mask0.sum() > 0:
                    sign_loss[mask0] = self.g_neutral * torch.abs(sign_overlap[mask0])
                # Penalize class 1 (targets==1) with relu(thresh - overlap)
                mask1 = (targets == 1)
                if mask1.sum() > 0:
                    sign_loss[mask1] = F.relu(self.thresh - sign_overlap[mask1])
            else:
                sign_loss = F.relu(self.thresh - sign_overlap)

        # Combine safely
        loss = ((1 - self.alpha) * bce_loss + self.alpha * safe_mean(sign_loss))

        return loss
