"""Copy a few test-set patients into examples/ as compressed NIfTI for the app.

    python -m src.make_examples

Test-set patients were never used for training or model selection. Picks a
large high-grade tumor, a low-grade tumor and the smallest tumor, so the demo
shows the range of difficulty.
"""
import json

import nibabel as nib
import numpy as np
import pandas as pd

from src.brats_io import BraTSSource
from src.utils import ROOT, SEQUENCES, load_config


def pick(pat, test):
    t = pat[pat.pid.isin(test)].sort_values("vol_WT_ml")
    hgg, lgg = t[t.grade == "HGG"], t[t.grade == "LGG"]
    return [hgg.iloc[int(len(hgg) * 0.85)].pid, lgg.iloc[len(lgg) // 2].pid, t.iloc[0].pid]


def main():
    cfg = load_config()
    pat = pd.read_csv(ROOT / cfg["processed_dir"] / "patients.csv")
    test = json.loads((ROOT / cfg["splits_file"]).read_text())["test"]
    source = BraTSSource(ROOT / cfg["raw_path"])
    info = []
    for pid in pick(pat, test):
        found, _ = source.locate(pid)
        out = ROOT / "examples" / pid
        out.mkdir(parents=True, exist_ok=True)
        for key in SEQUENCES + ["seg"]:
            img = source.load_nifti(found[key])
            data = np.rint(np.asarray(img.dataobj)).astype(np.uint8 if key == "seg" else np.int16)
            nii = nib.Nifti1Image(data, img.affine)
            nii.set_data_dtype(data.dtype)
            nib.save(nii, out / f"{pid}_{key}.nii.gz")
        p = pat[pat.pid == pid].iloc[0]
        info.append({"pid": pid, "grade": p.grade, "wt_ml": round(p.vol_WT_ml, 1)})
        print(pid, p.grade, f"WT {p.vol_WT_ml:.0f} mL",
              f"{sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB")
    (ROOT / "examples" / "examples.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
