# Progress

## Step 1 — Data ✅

Done:
- Downloaded BraTS 2020 via the Kaggle API (`awsaf49/brats20-dataset-training-validation`, 4.2 GB zip). Only the 369 labelled training patients are used.
- Verified on all 369 patients: 240×240×155, 1 mm isotropic, identical affine across the 4 sequences + label (co-registered), skull-stripped (zero background, visible in `reports/audit/samples.png`), labels ⊆ {0,1,2,4}.
- Issue found and handled: patient 355's label file is misnamed (`W39_1998.09.19_Segm.nii`). No patient skipped.
- Patient-level 70/15/15 split stratified by HGG/LGG, saved to `configs/splits.json` (258 / 55 / 56 patients); no overlap.
- Preprocessing (`src/preprocess.py`): per-sequence z-score on brain voxels, 192 px crop around the brain → 160×160, nested WT/TC/ET masks, cached as float16 `.npy` (11 GB, 6 min on 4 cores).
- Audit (`src/audit.py`, `notebooks/01_data_audit.ipynb`, `reports/audit/`).
- Dataset + augmentation + modality dropout (`src/data.py`), ~520 slices/s single process.

Issues / notes:
- Fixed: albumentations `RandomBrightnessContrast` clips floats to [0,1]; replaced with a hand-written per-sequence jitter.
- This work ran in a cloud container (CPU only). The code also runs on the Mac (`raw_path` can be the zip or an extracted folder).

## Step 2 — Models ✅

Done:
- `src/models.py`: plain U-Net (7.8M), Attention U-Net (7.9M), ResNet34 U-Net via smp (24.4M, ImageNet encoder, 4-channel input).
- `src/losses.py`: Dice + BCE loss; per-patient validation Dice.
- `src/train.py`: AdamW, LR halving on plateau, early stopping (patience 6, max 30 epochs), best/last checkpoints, `history.csv`, TensorBoard, `--resume`, `--estimate`.
- Runs in `configs/runs/`: `unet`, `attunet`, `resunet34` (all with modality dropout 0.3) and `resunet34_nomd` (control without it).
- Fixed: each DataLoader worker now gets its own augmentation seed.
- Smoke-tested on CPU: all models train, checkpoint and resume.


## Steps 3–5 — Evaluation, advanced features, app ✅

Done:
- `src/inference.py`: whole-scan prediction at native resolution, TTA (4 views), uncertainty map, missing sequences.
- `src/metrics.py`: 3D Dice, HD95, volume (mL), uncertainty-vs-error (AUROC, error rate in the 1% most uncertain voxels).
- `src/evaluate.py`: test set once — every model ± TTA, missing-sequence test (with vs without modality dropout) → `reports/eval/<split>/per_patient.csv`.
- `src/report.py`: results tables, model comparison, missing-sequence chart, best/worst gallery with automatic failure reasons, uncertainty figure.
- `src/viz.py`: overlays, uncertainty map, tumor area per slice, plotly 3D view.
- `src/export.py`: fp16 weights (ResNet34 U-Net 47 MB), ONNX, speed benchmark (PyTorch vs ONNX Runtime).
- `app.py`: Streamlit demo (examples or upload, missing-sequence warning, model choice, TTA, slice slider, overlay toggles, uncertainty, volumes, area chart, 3D view, expert-label comparison, NIfTI download, disclaimer, citations).
- `examples/`: 3 held-out test patients (24 MB), `requirements.txt` (CPU-only torch, verified in a clean environment).
- `notebooks/02_results.ipynb` (runs after evaluation).
- Tested end to end with quick CPU-trained models (numbers not reported — those models are placeholders).

- Trained all four models in one unattended Kaggle T4 run (`kaggle/train_kaggle.py`, 4 h 43 min incl. preprocessing and evaluation); early stopping after 16–22 epochs.
- Test set (56 patients), ResNet34 U-Net (chosen by validation Dice 0.825): Dice WT 0.900 / TC 0.834 / ET 0.787; all four models within ~0.01 Dice.
- Modality dropout: without FLAIR WT Dice 0.834 vs 0.583; without T1ce ET Dice 0.342 vs 0.049.
- Uncertainty AUROC 0.97; TTA +0.001–0.007 Dice.
- Weights (`models/*.pt`, 15–47 MB), results (`reports/eval/test/`), training histories (`reports/training/`) committed; results notebook executed; README results and app screenshot added.

Issues met:
- macOS "Too many open files" on the first Mac run → loader now reads slices without keeping files open.
- Kaggle notebook runs need the new-style Kaggle API token; the run was pushed from the Mac.

## Step 6 — Finish ✅

- Deployed on Streamlit Community Cloud: https://brain-tumor-seg-ml.streamlit.app/ (checked: example patient segments with the same volumes as locally, no errors).
- README with problem, approach, results, screenshot, live URL, how to run, limitations, citations; `docs/decisions.md`, `docs/deployment.md`.

Optional next steps:
- `python -m src.export --benchmark` on the Mac for MPS vs CPU vs ONNX timings.
- Segment the non-enhancing compartment separately (the weakest cases are low-grade gliomas).
