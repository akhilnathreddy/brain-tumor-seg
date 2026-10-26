"""PyTorch dataset over the cached 2D slices, with augmentation and modality dropout."""
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import bits_to_regions
from src.utils import ROOT


def train_augment():
    """Geometric transforms for image + mask (albumentations)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),   # left/right brain flip is anatomically plausible
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
    ])


def intensity_jitter(x, rng, limit=0.1, p=0.5):
    """Per-sequence brightness/contrast on z-scored data: x*a + b inside the brain.

    albumentations' RandomBrightnessContrast clips float images to [0, 1], which
    would wipe out every negative z-score, so this is done by hand.
    """
    brain = (x != 0).any(0)
    for c in range(x.shape[0]):
        if rng.random() < p:
            a = 1 + rng.uniform(-limit, limit)
            b = rng.uniform(-limit, limit)
            x[c] = (x[c] * a + b) * brain
    return x


class BraTSSlices(Dataset):
    """Slices of one split.

    For training, every epoch uses all tumor slices plus a fresh random subset of
    tumor-free brain slices (call `resample()` at the start of each epoch).
    For validation, every cached slice is used.
    """

    def __init__(self, split, cfg, train=False, modality_dropout=0.0, seed=42):
        self.cfg, self.train = cfg, train
        self.modality_dropout = modality_dropout
        self.dir = ROOT / cfg["processed_dir"] / "slices"
        df = pd.read_csv(ROOT / cfg["processed_dir"] / "slices.csv")
        self.all = df[df.split == split].reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.aug = train_augment() if train else None
        self._arrays = {}
        self.resample()

    def resample(self):
        df = self.all
        if self.train:
            tumor = df[df.WT_px > 0]
            empty = df[df.WT_px == 0]
            n = int(round(len(empty) * self.cfg["empty_slice_fraction"]))
            pick = self.rng.choice(len(empty), size=n, replace=False)
            df = pd.concat([tumor, empty.iloc[pick]])
        self.items = df[["pid", "row"]].to_numpy()

    def _arr(self, pid):
        # memory-mapped, so opening is cheap and only touched slices are read
        if pid not in self._arrays:
            self._arrays[pid] = (np.load(self.dir / f"{pid}_img.npy", mmap_mode="r"),
                                 np.load(self.dir / f"{pid}_msk.npy", mmap_mode="r"))
        return self._arrays[pid]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pid, row = self.items[i]
        img, msk = self._arr(pid)
        x = np.asarray(img[row], dtype=np.float32)          # (4,H,W)
        y = bits_to_regions(np.asarray(msk[row]))           # (3,H,W)
        if self.train:
            x, y = self._augment(x, y)
            if self.modality_dropout > 0 and self.rng.random() < self.modality_dropout:
                x[self.rng.integers(x.shape[0])] = 0.0       # simulate a missing sequence
        return torch.from_numpy(x), torch.from_numpy(y)

    def _augment(self, x, y):
        out = self.aug(image=x.transpose(1, 2, 0), mask=y.transpose(1, 2, 0).astype(np.uint8))
        x, y = out["image"].transpose(2, 0, 1).copy(), out["mask"].transpose(2, 0, 1)
        x = intensity_jitter(x, self.rng)
        return x, y.astype(np.float32)
