"""Evaluate trained models on full 3D scans (test set by default — run it once, at the end).

    python -m src.evaluate                      # all trained models, test split
    python -m src.evaluate --split val          # during development
    python -m src.evaluate --max-patients 3     # quick check

For every patient the raw NIfTI is loaded once and run through every
configuration: each model with and without test-time augmentation, plus the
missing-sequence test (each sequence removed in turn) for the models trained with
and without modality dropout. Writes reports/eval/<split>/per_patient.csv;
src.report turns it into tables and figures.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.brats_io import BraTSSource, check_and_load
from src.inference import TorchPredictor, load_model, predict_volume, to_regions
from src.metrics import dice, hd95, uncertainty_vs_error, volume_ml
from src.preprocess import labels_to_regions
from src.utils import REGIONS, ROOT, SEQUENCES, get_device, load_config

MISSING_TEST_MODELS = ["resunet34", "resunet34_nomd"]


def available_models():
    return sorted(p.parent.name for p in (ROOT / "checkpoints").glob("*/best.pt"))


def make_configs(models, tta=True, missing=True):
    configs = []
    for m in models:
        configs.append({"model": m, "tta": False, "drop": None})
        if tta:
            configs.append({"model": m, "tta": True, "drop": None})
    if missing:
        for m in [m for m in MISSING_TEST_MODELS if m in models]:
            for s in SEQUENCES:
                configs.append({"model": m, "tta": False, "drop": s})
    return configs


def score(out, gt):
    reg = to_regions(out["probs"].astype(np.float32))
    row = {}
    for i, r in enumerate(REGIONS):
        row[f"dice_{r}"] = dice(reg[i], gt[i])
        row[f"hd95_{r}"] = hd95(reg[i], gt[i])
        row[f"vol_pred_{r}"] = volume_ml(reg[i])
        row[f"vol_gt_{r}"] = volume_ml(gt[i])
        row[f"vol_err_{r}"] = row[f"vol_pred_{r}"] - row[f"vol_gt_{r}"]
    u = uncertainty_vs_error(out["uncertainty"], reg[0], gt[0], out["brain"])
    row.update({f"{k}_WT": v for k, v in u.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--models", nargs="*", help="default: every checkpoints/*/best.pt")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-missing", action="store_true")
    ap.add_argument("--max-patients", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    models = args.models or available_models()
    if not models:
        raise SystemExit("no trained models in checkpoints/ — train first")
    configs = make_configs(models, tta=not args.no_tta, missing=not args.no_missing)
    predictors = {m: TorchPredictor(load_model(ROOT / "checkpoints" / m / "best.pt", device)[0], device)
                  for m in models}

    pids = json.loads((ROOT / cfg["splits_file"]).read_text())[args.split]
    if args.max_patients:
        pids = pids[:args.max_patients]
    grades = pd.read_csv(ROOT / cfg["processed_dir"] / "patients.csv").set_index("pid")["grade"]
    source = BraTSSource(ROOT / cfg["raw_path"])
    print(f"{args.split}: {len(pids)} patients × {len(configs)} configurations on {device}")

    out_dir = ROOT / "reports" / "eval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in tqdm(pids):
        vol, seg, _ = check_and_load(source, pid)
        gt = labels_to_regions(seg)
        for c in configs:
            v = vol.copy()
            if c["drop"]:
                v[SEQUENCES.index(c["drop"])] = 0
            t = time.time()
            out = predict_volume(predictors[c["model"]], v, cfg, tta=c["tta"])
            rows.append({"pid": pid, "grade": grades.get(pid), **c, "seconds": time.time() - t,
                         **score(out, gt)})
        pd.DataFrame(rows).to_csv(out_dir / "per_patient.csv", index=False)  # save as we go
    print(f"wrote {out_dir / 'per_patient.csv'} — next: python -m src.report --split {args.split}")


if __name__ == "__main__":
    main()
