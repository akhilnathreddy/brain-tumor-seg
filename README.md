# Brain tumor segmentation (BraTS 2020)

Work in progress. See `PROGRESS.md` for status and `docs/decisions.md` for the reasoning behind each choice.

## Reproduce the data step

```bash
pip install -r requirements-train.txt
# Kaggle API token in ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY
kaggle datasets download awsaf49/brats20-dataset-training-validation -p data/raw
python -m src.preprocess   # ~6 min on 4 cores, ~11 GB cache in data/processed
python -m src.audit        # reports/audit/
```

## Train

```bash
# estimate first (prints minutes per epoch and total), then train
python -m src.train --run configs/runs/resunet34.yaml --estimate
python -m src.train --run configs/runs/resunet34.yaml
tensorboard --logdir runs        # live curves
```

Runs: `unet`, `attunet`, `resunet34`, `resunet34_nomd` (no modality dropout).
Interrupted? Add `--resume`.

## Data credit

BraTS 2020 — Menze et al., IEEE TMI 2015; Bakas et al., Scientific Data 2017; Bakas et al., arXiv:1811.02629 (2018).
