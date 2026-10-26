# Progress

## Step 1 — Data ✅ (waiting for review at the STOP point)

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
- TODO for Step 2: re-seed each DataLoader worker's RNG (otherwise workers share augmentation randomness).

## Next — Step 2: models
Plain U-Net, ResNet34-encoder U-Net (smp, 4-channel input), Attention U-Net; Dice+BCE, modality dropout, early stopping on val Dice, TensorBoard. Estimate training time before each run.
