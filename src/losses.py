"""Loss and Dice metric for the 3 nested regions (WT, TC, ET)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE (pixel-wise, stable) + soft Dice (overlap, robust to class imbalance).

    Soft Dice is computed per region over the whole batch, so a batch of
    tumor-free slices doesn't produce a 0/0 Dice.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.eps = bce_weight, dice_weight, eps

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return self.bce_weight * bce + self.dice_weight * (1 - dice.mean())


class PatientDice:
    """Accumulates hard-prediction overlap per patient and region over slices,
    then reports per-patient Dice (a 2D-slice approximation of the 3D score).

    Empty prediction on empty ground truth counts as Dice 1 (e.g. LGG without ET).
    """

    def __init__(self, n_regions=3):
        self.stats = {}
        self.n = n_regions

    @torch.no_grad()
    def update(self, logits, target, pids, threshold=0.5):
        pred = (torch.sigmoid(logits) > threshold).float()
        inter = (pred * target).sum((2, 3)).cpu()
        psum = pred.sum((2, 3)).cpu()
        tsum = target.sum((2, 3)).cpu()
        for i, pid in enumerate(pids):
            s = self.stats.setdefault(pid, torch.zeros(3, self.n))
            s[0] += inter[i]
            s[1] += psum[i]
            s[2] += tsum[i]

    def per_patient(self):
        out = {}
        for pid, (inter, psum, tsum) in self.stats.items():
            denom = psum + tsum
            out[pid] = torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))
        return out

    def mean(self):
        d = torch.stack(list(self.per_patient().values()))
        return d.mean(0)  # (n_regions,)
