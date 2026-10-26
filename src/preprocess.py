"""Preprocessing: normalize -> crop around brain -> resize slices -> cache as .npy.

The geometry helpers here are shared with inference, so training data and app
inputs go through exactly the same steps.

Run:  python -m src.preprocess
"""
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.brats_io import BraTSSource, check_and_load
from src.utils import REGIONS, ROOT, SEQUENCES, load_config, set_seed

# ---------------------------------------------------------------- shared steps


def brain_mask(vol):
    """vol (C,X,Y,Z) -> bool (X,Y,Z): voxel is brain if any sequence is non-zero."""
    return (vol > 0).any(0)


def normalize(vol, mask, clip=(0.5, 99.5)):
    """Per-sequence z-score using brain voxels only; background stays 0.

    A sequence that is all zeros (missing) is left as zeros.
    """
    out = np.zeros_like(vol, dtype=np.float32)
    for c in range(vol.shape[0]):
        v = vol[c][mask]
        if v.size == 0 or v.max() <= 0:
            continue
        lo, hi = np.percentile(v, clip)
        v = np.clip(v, lo, hi)
        out[c][mask] = (v - v.mean()) / max(v.std(), 1e-6)
    return out


def crop_box(mask, min_side):
    """Square in-plane box centred on the brain: (x0, y0, side)."""
    xs = np.where(mask.any((1, 2)))[0]
    ys = np.where(mask.any((0, 2)))[0]
    if xs.size == 0:  # no brain at all: use the image centre
        X, Y = mask.shape[:2]
        return X // 2 - min_side // 2, Y // 2 - min_side // 2, min_side
    side = max(min_side, int(np.ptp(xs)) + 9, int(np.ptp(ys)) + 9)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    return int(cx - side // 2), int(cy - side // 2), int(side)


def crop(arr, x0, y0, side):
    """arr (..., X, Y, Z) -> (..., side, side, Z), zero-padded where the box leaves the image."""
    X, Y = arr.shape[-3], arr.shape[-2]
    out = np.zeros(arr.shape[:-3] + (side, side, arr.shape[-1]), dtype=arr.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, X), min(y0 + side, Y)
    out[..., sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0, :] = arr[..., sx0:sx1, sy0:sy1, :]
    return out


def uncrop(arr, x0, y0, shape_xy):
    """Inverse of crop: (..., side, side, Z) -> (..., X, Y, Z)."""
    side = arr.shape[-2]
    X, Y = shape_xy
    out = np.zeros(arr.shape[:-3] + (X, Y, arr.shape[-1]), dtype=arr.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, X), min(y0 + side, Y)
    out[..., sx0:sx1, sy0:sy1, :] = arr[..., sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0, :]
    return out


def resize_slices(arr, size):
    """arr (C, S, S, Z) float -> (Z, C, size, size) float32 (bilinear, antialiased)."""
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(3, 0, 1, 2))).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False,
                      antialias=t.shape[-1] > size)
    return t.numpy()


def labels_to_regions(seg):
    """BraTS labels (1 necrotic core, 2 edema, 4 enhancing) -> nested bool regions (3, ...)."""
    wt = seg > 0
    tc = (seg == 1) | (seg == 4)
    et = seg == 4
    return np.stack([wt, tc, et])


def regions_to_bits(reg):
    """(3, ...) bool -> uint8 bitmask (bit0 WT, bit1 TC, bit2 ET)."""
    return (reg[0] * 1 + reg[1] * 2 + reg[2] * 4).astype(np.uint8)


def bits_to_regions(bits):
    return np.stack([(bits >> i) & 1 for i in range(3)]).astype(np.float32)


def prepare_volume(vol, cfg):
    """Raw (4,X,Y,Z) -> (normalized cropped volume (4,S,S,Z), crop box, brain mask)."""
    mask = brain_mask(vol)
    norm = normalize(vol, mask, tuple(cfg["clip_percentiles"]))
    x0, y0, side = crop_box(mask, cfg["min_crop"])
    return crop(norm, x0, y0, side), (x0, y0, side), mask


# ---------------------------------------------------------------- caching

_source = None


def _init_worker(raw_path):
    global _source
    _source = BraTSSource(ROOT / raw_path)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def process_patient(args):
    pid, split, grade, cfg = args
    vol, seg, info = check_and_load(_source, pid)
    row = {"pid": pid, "grade": grade, "split": split, "status": "ok",
           "issues": "; ".join(info["issues"])}
    if not info["ok"]:
        row["status"] = "skipped"
        return row, []

    size = cfg["img_size"]
    cropped, (x0, y0, side), mask = prepare_volume(vol, cfg)
    regions = labels_to_regions(seg)                    # (3,X,Y,Z) native
    reg_crop = crop(regions.astype(np.float32), x0, y0, side)

    brain_px = mask.sum((0, 1))
    reg_px = regions.sum((1, 2))                         # (3, Z)
    keep = np.where((brain_px >= cfg["min_brain_pixels"]) | (reg_px[0] > 0))[0]

    img = resize_slices(cropped[..., keep], size).astype(np.float16)
    reg_small = resize_slices(reg_crop[..., keep], size) > 0.5   # nested regions stay nested
    bits = regions_to_bits(reg_small.transpose(1, 0, 2, 3))
    out = ROOT / cfg["processed_dir"] / "slices"
    np.save(out / f"{pid}_img.npy", img)
    np.save(out / f"{pid}_msk.npy", bits)

    # resolution ceiling: GT -> 160px -> back to native, Dice against GT
    back = np.zeros((3, side, side, seg.shape[2]), dtype=np.float32)
    back[..., keep] = resize_slices(reg_small.transpose(1, 2, 3, 0).astype(np.float32),
                                    side).transpose(1, 2, 3, 0)
    back = uncrop(back, x0, y0, seg.shape[:2]) > 0.5

    voxel_ml = float(np.prod(info["zooms"])) / 1000.0
    brain_xy = np.where(mask.any(2))
    row.update({
        "x0": x0, "y0": y0, "side": side, "n_slices": len(keep),
        "n_tumor_slices": int((reg_px[0][keep] > 0).sum()),
        "brain_dx": int(np.ptp(brain_xy[0]) + 1), "brain_dy": int(np.ptp(brain_xy[1]) + 1),
        "brain_ml": float(mask.sum() * voxel_ml),
        "labels": ",".join(map(str, info["labels"])),
        "img_dtypes": ",".join(info["img_dtypes"]), "seg_dtype": info["seg_dtype"],
        "zooms": "x".join(f"{z:g}" for z in info["zooms"]),
        "shape": "x".join(map(str, info["shape"])),
        "tumor_outside_brain_vox": int((regions[0] & ~mask).sum()),
    })
    for r, name in enumerate(REGIONS):
        row[f"vol_{name}_ml"] = float(regions[r].sum() * voxel_ml)
        row[f"ceiling_dice_{name}"] = float(dice(back[r], regions[r]))
    slices = [{"pid": pid, "split": split, "row": i, "z": int(z), "brain_px": int(brain_px[z]),
               **{f"{n}_px": int(reg_px[r][z]) for r, n in enumerate(REGIONS)}}
              for i, z in enumerate(keep)]
    return row, slices


def make_splits(pids, grades, cfg):
    """70/15/15 by patient, stratified by HGG/LGG."""
    s = cfg["split"]
    strat = [grades.get(p, "NA") for p in pids]
    train, rest, _, strat_rest = train_test_split(
        pids, strat, train_size=s["train"], stratify=strat, random_state=cfg["seed"])
    val, test = train_test_split(
        rest, train_size=s["val"] / (s["val"] + s["test"]), stratify=strat_rest,
        random_state=cfg["seed"])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    source = BraTSSource(ROOT / cfg["raw_path"])
    pids, grades = source.patient_ids(), source.grades()
    print(f"found {len(pids)} patient folders; {len(grades)} grades in name_mapping.csv")

    splits_path = ROOT / cfg["splits_file"]
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        print(f"using existing splits from {cfg['splits_file']}")
    else:
        splits = make_splits(pids, grades, cfg)
        splits_path.write_text(json.dumps(splits, indent=1))
    split_of = {p: k for k, v in splits.items() for p in v}
    assert len(split_of) == sum(len(v) for v in splits.values()), "patient in two splits"

    out = ROOT / cfg["processed_dir"]
    (out / "slices").mkdir(parents=True, exist_ok=True)
    jobs = [(p, split_of[p], grades.get(p, "NA"), cfg) for p in pids]
    rows, slices, t0 = [], [], time.time()
    with Pool(4, initializer=_init_worker, initargs=(cfg["raw_path"],)) as pool:
        for i, (row, sl) in enumerate(pool.imap_unordered(process_patient, jobs), 1):
            rows.append(row)
            slices += sl
            if row["issues"]:
                print(f"  {row['pid']}: {row['issues']}")
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} patients, {time.time() - t0:.0f}s")
    pd.DataFrame(rows).sort_values("pid").to_csv(out / "patients.csv", index=False)
    pd.DataFrame(slices).sort_values(["pid", "row"]).to_csv(out / "slices.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
