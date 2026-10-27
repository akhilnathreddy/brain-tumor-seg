"""Whole-scan inference: 4 NIfTI files in, 3D tumor regions + confidence out.

Uses exactly the training preprocessing (src.preprocess), predicts every axial
slice that contains brain, and maps the result back to the original 240x240x155
grid, so volumes in mL and distances in mm are meaningful.
Shared by evaluation, ONNX benchmarking and the Streamlit app.
"""
import gzip
import io
import math

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.models import build_model
from src.preprocess import prepare_volume, resize_slices, uncrop
from src.utils import SEQUENCES

# Test-time augmentation views: (rotation in degrees, horizontal flip).
# All lie inside the training augmentation range, so the model has seen them.
TTA_VIEWS = [(0, False), (0, True), (10, False), (-10, False)]


# ---------------------------------------------------------------- loading


def read_nifti(src):
    """Path or raw bytes (.nii or .nii.gz) -> nibabel image."""
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return nib.Nifti1Image.from_bytes(data)
    return nib.load(str(src))


def load_sequences(files):
    """{seq: path or bytes} (any subset of the 4) -> (vol (4,X,Y,Z) float32, affine, missing).

    A missing sequence becomes all zeros, which is what the model saw during
    training with modality dropout.
    """
    imgs = {s: read_nifti(files[s]) for s in SEQUENCES if files.get(s) is not None}
    if not imgs:
        raise ValueError("no MRI sequence given")
    ref = next(iter(imgs.values()))
    for s, im in imgs.items():
        if im.shape != ref.shape:
            raise ValueError(f"{s} has shape {im.shape}, expected {ref.shape} like the others")
    vol = np.zeros((4,) + ref.shape[:3], dtype=np.float32)
    for i, s in enumerate(SEQUENCES):
        if s in imgs:
            vol[i] = np.nan_to_num(np.asarray(imgs[s].dataobj, dtype=np.float32))
    missing = [s for s in SEQUENCES if s not in imgs]
    return vol, ref.affine, missing


def load_model(path, device="cpu"):
    """Load a training checkpoint (best.pt) or an exported half-precision model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], pretrained=False)
    state = ckpt.get("model") or ckpt["state_dict"]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(device).eval(), ckpt


class TorchPredictor:
    """Callable: (B,4,H,W) float tensor -> (B,3,H,W) logits."""

    def __init__(self, model, device):
        self.model, self.device = model, torch.device(device)

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x.to(self.device)).float().cpu()


class OnnxPredictor:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, x):
        return torch.from_numpy(self.sess.run(None, {self.name: x.numpy()})[0])


# ---------------------------------------------------------------- TTA


def _warp(x, angle, flip):
    """Rotate (degrees) and optionally flip a (B,C,H,W) batch."""
    if flip:
        x = x.flip(-1)
    if angle:
        a = math.radians(angle)
        theta = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]],
                             dtype=x.dtype).expand(x.shape[0], 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return x


def _unwarp(x, angle, flip):
    x = _warp(x, -angle, False)
    return x.flip(-1) if flip else x


# ---------------------------------------------------------------- prediction


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def predict_volume(predictor, vol, cfg, tta=False, batch_size=32):
    """Raw (4,X,Y,Z) volume -> dict with native-resolution float16 arrays:

    probs (3,X,Y,Z): mean sigmoid probability for WT, TC, ET
    uncertainty (X,Y,Z): binary entropy of the mean probability, max over regions
                         (0 = certain, 1 = coin flip). With TTA the mean also
                         includes disagreement between the augmented views.
    """
    size = cfg["img_size"]
    cropped, (x0, y0, side), mask = prepare_volume(vol, cfg)
    X, Y, Z = vol.shape[1:]
    zs = np.where(mask.any((0, 1)))[0]                      # slices containing brain
    slices = torch.from_numpy(resize_slices(cropped[..., zs], size))   # (n,4,size,size)
    views = TTA_VIEWS if tta else TTA_VIEWS[:1]

    out = np.zeros((len(zs), 3, size, size), dtype=np.float32)
    for i in range(0, len(zs), batch_size):
        x = slices[i:i + batch_size]
        acc = torch.zeros(x.shape[0], 3, size, size)
        for angle, flip in views:
            acc += _unwarp(torch.sigmoid(predictor(_warp(x, angle, flip))), angle, flip)
        out[i:i + batch_size] = (acc / len(views)).numpy()

    # back to native resolution: resize to the crop size, paste into the full grid
    probs_crop = np.zeros((3, side, side, Z), dtype=np.float32)
    if len(zs):
        probs_crop[..., zs] = resize_slices(out.transpose(1, 2, 3, 0), side).transpose(1, 2, 3, 0)
    probs = np.clip(uncrop(probs_crop, x0, y0, (X, Y)), 0, 1)
    probs[:, ~mask] = 0                                      # nothing outside the brain
    unc = binary_entropy(probs).max(0)
    unc[~mask] = 0
    return {"probs": probs.astype(np.float16), "uncertainty": unc.astype(np.float16),
            "brain": mask}


def to_regions(probs, threshold=0.5):
    """Probabilities -> nested boolean masks (3,...) with ET ⊆ TC ⊆ WT enforced."""
    wt = probs[0] > threshold
    tc = (probs[1] > threshold) & wt
    et = (probs[2] > threshold) & tc
    return np.stack([wt, tc, et])


def regions_to_labels(reg):
    """Nested regions -> BraTS labels (2 edema, 1 necrotic core, 4 enhancing)."""
    lab = np.zeros(reg.shape[1:], dtype=np.uint8)
    lab[reg[0]] = 2
    lab[reg[1]] = 1
    lab[reg[2]] = 4
    return lab
