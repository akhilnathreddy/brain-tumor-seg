"""Reading BraTS 2020 training data from the Kaggle zip or an extracted folder.

Reading straight from the zip means the ~30 GB of uncompressed NIfTI never has to
be extracted. Both layouts are handled the same way: we index every file by
patient folder, then pick out the 4 sequences and the segmentation.
"""
import io
import re
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.utils import SEQUENCES

PID_RE = re.compile(r"(BraTS20_Training_\d{3})/([^/]+)$")


class BraTSSource:
    def __init__(self, path):
        self.path = Path(path)
        self.is_zip = self.path.suffix == ".zip"
        self._zip = None
        self.files = {}          # pid -> {filename: member/path}
        self.name_mapping = None
        names = self._list_names()
        for n in names:
            m = PID_RE.search(n)
            if m and n.endswith((".nii", ".nii.gz")):
                self.files.setdefault(m.group(1), {})[m.group(2)] = n
            elif n.endswith("name_mapping.csv") and "Training" in n:
                self.name_mapping = n

    def _list_names(self):
        if self.is_zip:
            return self.zip.namelist()
        return [str(p) for p in self.path.rglob("*") if p.is_file()]

    @property
    def zip(self):
        # opened lazily so each worker process gets its own handle
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.path)
        return self._zip

    def _read_bytes(self, name):
        if self.is_zip:
            return self.zip.read(name)
        return Path(name).read_bytes()

    def patient_ids(self):
        return sorted(self.files)

    def grades(self):
        """pid -> 'HGG'/'LGG' from name_mapping.csv."""
        if self.name_mapping is None:
            return {}
        df = pd.read_csv(io.BytesIO(self._read_bytes(self.name_mapping)))
        return dict(zip(df["BraTS_2020_subject_ID"], df["Grade"]))

    def locate(self, pid):
        """Return ({seq: name, 'seg': name}, issues). Missing entries are absent."""
        found, issues = {}, []
        names = self.files.get(pid, {})
        for key in SEQUENCES + ["seg"]:
            for ext in (".nii", ".nii.gz"):
                fname = f"{pid}_{key}{ext}"
                if fname in names:
                    found[key] = names[fname]
        if "seg" not in found:
            # e.g. patient 355 ships its label as 'W39_1998.09.19_Segm.nii'
            cands = [f for f in names if "seg" in f.lower()
                     and not any(f.startswith(f"{pid}_{s}.") for s in SEQUENCES)]
            if len(cands) == 1:
                found["seg"] = names[cands[0]]
                issues.append(f"misnamed seg file '{cands[0]}' (used it)")
        for key in SEQUENCES + ["seg"]:
            if key not in found:
                issues.append(f"missing {key}")
        return found, issues

    def load_nifti(self, name):
        data = self._read_bytes(name)
        if name.endswith(".gz"):
            import gzip
            data = gzip.decompress(data)
        return nib.Nifti1Image.from_bytes(data)


def check_and_load(source, pid):
    """Load one patient and run sanity checks.

    Returns (images (4,X,Y,Z) float32, seg (X,Y,Z) uint8 or None, info dict).
    info['ok'] is False when the patient must be skipped.
    """
    found, issues = source.locate(pid)
    info = {"pid": pid, "issues": issues, "ok": True}
    if any(i.startswith("missing") for i in issues):
        info["ok"] = False
        return None, None, info

    imgs = {k: source.load_nifti(found[k]) for k in SEQUENCES + ["seg"]}
    ref = imgs["flair"]
    info["shape"] = ref.shape
    info["zooms"] = tuple(float(z) for z in ref.header.get_zooms()[:3])
    info["img_dtypes"] = sorted({str(imgs[s].get_data_dtype()) for s in SEQUENCES})
    info["seg_dtype"] = str(imgs["seg"].get_data_dtype())

    for k, im in imgs.items():
        if im.shape != ref.shape:
            issues.append(f"{k} shape {im.shape} != {ref.shape}")
            info["ok"] = False
        elif not np.allclose(im.affine, ref.affine, atol=1e-3):
            issues.append(f"{k} affine differs from flair")
    if not np.allclose(info["zooms"], 1.0, atol=1e-3):
        issues.append(f"voxel size {info['zooms']} is not 1 mm")
    if not info["ok"]:
        return None, None, info

    vol = np.stack([np.asarray(imgs[s].dataobj, dtype=np.float32) for s in SEQUENCES])
    if not np.isfinite(vol).all():
        issues.append("non-finite values (set to 0)")
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    for i, s in enumerate(SEQUENCES):
        if not (vol[i] > 0).any():
            issues.append(f"{s} is empty")

    seg_raw = np.asarray(imgs["seg"].dataobj)
    seg = np.rint(seg_raw).astype(np.int16)
    labels = set(np.unique(seg).tolist())
    if not labels <= {0, 1, 2, 4}:
        issues.append(f"unexpected labels {sorted(labels - {0, 1, 2, 4})} (set to 0)")
        seg[~np.isin(seg, [0, 1, 2, 4])] = 0
    info["labels"] = sorted(labels)
    return vol, seg.astype(np.uint8), info
