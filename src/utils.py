import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ["flair", "t1", "t1ce", "t2"]
REGIONS = ["WT", "TC", "ET"]
# One fixed color per region, used in every figure and in the app.
REGION_COLORS = {"WT": "#2a78d6", "TC": "#eb6834", "ET": "#1baf7a"}


def load_config(path="configs/data.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device():
    """MPS on Apple Silicon, then CUDA, then CPU."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
