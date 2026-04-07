import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_mean(tensor, *, device=None, dtype=None):
    if tensor.numel() == 0:
        return torch.tensor(
            0.0,
            device=device or tensor.device,
            dtype=dtype or tensor.dtype,
        )
    return tensor.mean()


class BCEOneClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, class_bal=1):
        targets = targets.float()
        bce = self.criterion(logits, targets)

        if class_bal != 0:
            return safe_mean(bce)

        mask_pos = targets == 1
        mask_neg = targets == 0

        pos_loss = safe_mean(
            bce[mask_pos],
            device=logits.device,
            dtype=logits.dtype,
        )

        neutral_targets = torch.full_like(logits[mask_neg], 0.5)
        neutral_loss = safe_mean(
            self.criterion(logits[mask_neg], neutral_targets),
            device=logits.device,
            dtype=logits.dtype,
        )

        return pos_loss + 0.01 * neutral_loss


class SignBCELoss(nn.Module):
    def __init__(self, alpha=0.5, thresh=1.0):
        super().__init__()
        self.alpha = alpha
        self.thresh = thresh
        self.bce_with_logits = BCEOneClassLoss()

    def forward(self, readout, targets, class_bal=1):
        zero = torch.tensor(0.0, device=readout.device, dtype=readout.dtype)

        bce_loss = (
            self.bce_with_logits(readout, targets, class_bal=class_bal)
            if self.alpha != 1.0
            else zero
        )

        sign_overlap = torch.sign(2 * targets - 1) * readout

        if self.alpha == 0:
            sign_loss = zero
        elif class_bal == 0:
            mask0 = targets == 0
            mask1 = targets == 1
            loss0 = 0.1 * safe_mean(
                torch.abs(sign_overlap[mask0]),
                device=readout.device,
                dtype=readout.dtype,
            )
            loss1 = safe_mean(
                F.relu(self.thresh - sign_overlap[mask1]),
                device=readout.device,
                dtype=readout.dtype,
            )
            sign_loss = loss0 + loss1
        else:
            sign_loss = safe_mean(
                F.relu(self.thresh - sign_overlap),
                device=readout.device,
                dtype=readout.dtype,
            )

        return (1 - self.alpha) * bce_loss + self.alpha * sign_loss


class DualLoss(nn.Module):
    def __init__(
        self,
        device,
        alpha=1.0,
        thresh=5.0,
        stim_idx=None,
        gng_idx=None,
        cue_idx=None,
        test_idx=None,
        rwd_idx=None,
        zero_idx=None,
        read_idx=None,
        class_bal=None,
    ):
        super().__init__()

        self.class_bal = [0] if class_bal is None else list(class_bal)
        self.read_idx = [-1] if read_idx is None else list(read_idx)
        self.zero_idx = [] if zero_idx is None else list(zero_idx)

        self.stim_idx = torch.as_tensor(
            [] if stim_idx is None else stim_idx,
            dtype=torch.long,
            device=device,
        )
        self.gng_idx = torch.as_tensor(
            [] if gng_idx is None else gng_idx,
            dtype=torch.long,
            device=device,
        )
        self.cue_idx = torch.as_tensor(
            [] if cue_idx is None else cue_idx,
            dtype=torch.long,
            device=device,
        )
        self.test_idx = torch.as_tensor(
            [] if test_idx is None else test_idx,
            dtype=torch.long,
            device=device,
        )
        self.rwd_idx = torch.as_tensor(
            [] if rwd_idx is None else rwd_idx,
            dtype=torch.long,
            device=device,
        )

        self.loss = SignBCELoss(alpha=alpha, thresh=thresh)
        self.l1loss = nn.SmoothL1Loss()

    def _segment_loss(self, readout, targets, idx, read_idx, target_channel, class_bal):
        if idx.numel() == 0:
            return torch.tensor(0.0, device=readout.device, dtype=readout.dtype)

        return self.loss(
            readout[:, idx, read_idx],
            targets[:, target_channel, : idx.numel()],
            class_bal=class_bal,
        )

    def _zero_penalty(self, readout):
        if not self.zero_idx:
            return torch.tensor(0.0, device=readout.device, dtype=readout.dtype)

        loss = torch.tensor(0.0, device=readout.device, dtype=readout.dtype)

        zero_slice = readout[:, self.zero_idx, self.read_idx[0]]
        loss = loss + self.l1loss(zero_slice, torch.zeros_like(zero_slice))

        if len(self.read_idx) > 1 and self.stim_idx.numel() > 0:
            stim0 = int(self.stim_idx[0].item())
            if stim0 > 1:
                pre_stim = readout[:, : stim0 - 1, self.read_idx[1]]
                loss = loss + self.l1loss(pre_stim, torch.zeros_like(pre_stim))

        return loss

    def _segment_specs(self):
        return [
            (
                self.test_idx,
                self.read_idx[-1],
                -1,
                1,
            ),
            (
                self.cue_idx,
                self.read_idx[3] if len(self.read_idx) > 3 else self.read_idx[-1],
                2,
                self.class_bal[3] if len(self.class_bal) > 3 else 1,
            ),
            (
                self.gng_idx,
                self.read_idx[2] if len(self.read_idx) > 2 else self.read_idx[-1],
                2,
                1,
            ),
            (
                self.stim_idx,
                self.read_idx[1] if len(self.read_idx) > 1 else self.read_idx[-1],
                1,
                1,
            ),
            (
                self.rwd_idx,
                self.read_idx[0],
                0,
                self.class_bal[0] if len(self.class_bal) > 0 else 1,
            ),
        ]

    def forward(self, readout, targets):
        loss = self._zero_penalty(readout)

        for idx, read_idx, target_channel, class_bal in self._segment_specs():
            loss = loss + self._segment_loss(
                readout,
                targets,
                idx,
                read_idx,
                target_channel,
                class_bal,
            )

        return loss
