"""Kaggle GPU run for brain-tumor-seg.

Clones the GitHub repo, preprocesses the attached BraTS 2020 dataset, trains all
four models, then evaluates on the test set, builds the report and exports the
half-precision weights. Everything worth keeping is copied to /kaggle/working/out.
"""
import resource
import shutil
import subprocess
import time
from pathlib import Path

import yaml

RUNS = ["resunet34", "resunet34_nomd", "unet", "attunet"]
WORK = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO, CACHE, OUT = WORK / "repo", WORK / "processed", Path("/kaggle/working/out")


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    code = subprocess.run(cmd, shell=True, cwd=REPO if REPO.exists() else None).returncode
    print(f"[exit {code} after {(time.time() - t) / 60:.1f} min]", flush=True)
    if check and code:
        raise SystemExit(code)
    return code


def keep(src, dst):
    """Copy a file or folder into the output directory if it exists."""
    src, dst = REPO / src, OUT / dst
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# the data loader memory-maps two cache files per patient in every worker
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else 65536, hard))

OUT.mkdir(parents=True, exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"df -h /kaggle/working {WORK}")
sh(f"git clone --depth 1 https://github.com/akhilnathreddy/brain-tumor-seg {REPO}")
sh('pip install -q "albumentations>=2.0.5" segmentation-models-pytorch nibabel onnx onnxruntime scikit-image plotly')
sh('python -c "import torch, albumentations as A, segmentation_models_pytorch as smp; '
   'print(torch.__version__, torch.cuda.is_available(), A.__version__, smp.__version__)"')

# point the data config at the attached dataset, cache outside /kaggle/working
raw = next(p for p in Path("/kaggle/input").glob("**/MICCAI_BraTS2020_TrainingData") if p.is_dir())
cfg_file = REPO / "configs" / "data.yaml"
cfg = yaml.safe_load(cfg_file.read_text())
cfg.update(raw_path=str(raw), processed_dir=str(CACHE))
cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("raw data:", raw, flush=True)

sh("python -m src.preprocess")
shutil.copy2(CACHE / "patients.csv", OUT / "patients.csv")

for run in RUNS:
    sh(f"python -m src.train --run configs/runs/{run}.yaml", check=False)
    for f in ["best.pt", "history.csv", "config.yaml"]:
        keep(f"checkpoints/{run}/{f}", f"checkpoints/{run}/{f}")
    keep(f"runs/{run}", f"runs/{run}")

# evaluation code was pushed to GitHub while training ran
sh("git pull --ff-only", check=False)
sh("python -m src.evaluate", check=False)
sh("python -m src.report", check=False)
keep("reports/eval/test", "reports/eval/test")
sh("python -m src.export", check=False)
for run in RUNS:
    keep(f"models/{run}.pt", f"models/{run}.pt")
sh(f"du -sh {OUT}; find {OUT} -maxdepth 3 | sort")
print("ALL DONE", flush=True)
