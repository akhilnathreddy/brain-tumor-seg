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

    def __init__(self, split, cfg, train=False, modality_dropout=0.0, seed=42, return_pid=False):
        self.cfg, self.train, self.return_pid = cfg, train, return_pid
        self.modality_dropout = modality_dropout
        self.dir = ROOT / cfg["processed_dir"] / "slices"
        df = pd.read_csv(ROOT / cfg["processed_dir"] / "slices.csv")
        self.all = df[df.split == split].reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.aug = train_augment() if train else None
        self._layout = {}
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

    def _read(self, pid, kind, row):
        """Read one slice from a cached .npy file without keeping the file open.

        Memory-mapping every patient keeps hundreds of files open per worker, more
        than macOS allows by default (256), so each read opens, seeks and closes.
        """
        path = self.dir / f"{pid}_{kind}.npy"
        if path not in self._layout:
            with open(path, "rb") as f:
                version = np.lib.format.read_magic(f)
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, _, dtype = read_header(f)
                self._layout[path] = (f.tell(), shape[1:], dtype)
        offset, shape, dtype = self._layout[path]
        nbytes = int(np.prod(shape)) * dtype.itemsize
        with open(path, "rb") as f:
            f.seek(offset + row * nbytes)
            return np.frombuffer(f.read(nbytes), dtype=dtype).reshape(shape)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pid, row = self.items[i]
        x = self._read(pid, "img", row).astype(np.float32)  # (4,H,W)
        y = bits_to_regions(self._read(pid, "msk", row))     # (3,H,W)
        if self.train:
            x, y = self._augment(x, y)
            if self.modality_dropout > 0 and self.rng.random() < self.modality_dropout:
                x[self.rng.integers(x.shape[0])] = 0.0       # simulate a missing sequence
        if self.return_pid:
            return torch.from_numpy(x), torch.from_numpy(y), pid
        return torch.from_numpy(x), torch.from_numpy(y)

    def _augment(self, x, y):
        out = self.aug(image=x.transpose(1, 2, 0), mask=y.transpose(1, 2, 0).astype(np.uint8))
        x, y = out["image"].transpose(2, 0, 1).copy(), out["mask"].transpose(2, 0, 1)
        x = intensity_jitter(x, self.rng)
        return x, y.astype(np.float32)


def worker_init_fn(worker_id):
    """Give each DataLoader worker its own augmentation RNG.

    torch hands every worker a different seed each epoch (derived from the main
    seed), so runs stay reproducible but workers don't repeat each other.
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)
