# Results — test set (56 patients)

Best model by validation Dice: **ResNet34 U-Net** (val mean Dice 0.825). Values are mean (median) over patients.

## Model comparison (no TTA)

| model | Dice WT | Dice TC | Dice ET | HD95 WT (mm) | HD95 TC (mm) | HD95 ET (mm) | \|vol err\| WT (mL) | \|vol err\| TC (mL) | \|vol err\| ET (mL) |
|---|---|---|---|---|---|---|---|---|---|
| Attention U-Net | 0.906 (0.933) | 0.829 (0.901) | 0.783 (0.861) | 7.7 (3.2) | 8.2 (5.4) | 27.3 (2.1) | 7.6 (4.8) | 6.2 (2.0) | 2.4 (1.6) |
| ResNet34 U-Net | 0.900 (0.920) | 0.834 (0.906) | 0.787 (0.864) | 19.6 (6.0) | 8.1 (3.3) | 24.6 (2.0) | 7.6 (4.4) | 6.0 (1.8) | 2.5 (1.4) |
| ResNet34 U-Net (no modality dropout) | 0.903 (0.930) | 0.835 (0.903) | 0.760 (0.860) | 12.8 (4.0) | 8.8 (5.2) | 32.0 (2.1) | 6.3 (2.9) | 5.7 (2.2) | 2.6 (1.5) |
| U-Net | 0.902 (0.924) | 0.826 (0.880) | 0.793 (0.868) | 8.7 (3.7) | 9.2 (6.4) | 27.2 (2.2) | 7.7 (5.9) | 5.7 (2.1) | 2.3 (1.2) |

## Test-time augmentation

| model | Dice WT: plain → TTA | Dice TC: plain → TTA | Dice ET: plain → TTA | seconds/patient: plain → TTA |
|---|---|---|---|---|
| Attention U-Net | 0.906 → 0.909 (+0.002) | 0.829 → 0.835 (+0.006) | 0.783 → 0.791 (+0.007) | 2.4 → 3.7 |
| ResNet34 U-Net | 0.900 → 0.903 (+0.004) | 0.834 → 0.839 (+0.005) | 0.787 → 0.791 (+0.004) | 2.2 → 3.2 |
| ResNet34 U-Net (no modality dropout) | 0.903 → 0.905 (+0.002) | 0.835 → 0.843 (+0.007) | 0.760 → 0.783 (+0.023) | 2.2 → 3.2 |
| U-Net | 0.902 → 0.903 (+0.001) | 0.826 → 0.832 (+0.006) | 0.793 → 0.796 (+0.003) | 2.3 → 3.4 |

## Missing-sequence test (mean Dice WT / TC / ET)

| removed sequence | ResNet34 U-Net: WT / TC / ET | ResNet34 U-Net (no modality dropout): WT / TC / ET |
|---|---|---|
| — (all 4) | 0.900 / 0.834 / 0.787 | 0.903 / 0.835 / 0.760 |
| FLAIR | 0.834 / 0.800 / 0.765 | 0.583 / 0.746 / 0.717 |
| T1 | 0.893 / 0.801 / 0.772 | 0.885 / 0.789 / 0.743 |
| T1CE | 0.895 / 0.658 / 0.342 | 0.887 / 0.412 / 0.049 |
| T2 | 0.889 / 0.822 / 0.783 | 0.875 / 0.784 / 0.771 |

## Does uncertainty point at errors? (WT)

| model | TTA | AUROC (uncertainty → error) | error rate, 1% most uncertain voxels | error rate, other voxels |
|---|---|---|---|---|
| Attention U-Net | no | 0.975 | 28.2% | 0.73% |
| Attention U-Net | yes | 0.975 | 27.1% | 0.71% |
| ResNet34 U-Net | no | 0.972 | 30.0% | 0.79% |
| ResNet34 U-Net | yes | 0.974 | 29.1% | 0.76% |
| ResNet34 U-Net (no modality dropout) | no | 0.973 | 28.6% | 0.72% |
| ResNet34 U-Net (no modality dropout) | yes | 0.975 | 27.8% | 0.70% |
| U-Net | no | 0.973 | 29.5% | 0.77% |
| U-Net | yes | 0.974 | 28.9% | 0.77% |

## Why the worst patients fail

- **BraTS20_Training_266**: low-grade glioma (under-represented: ~20% of training data); tiny true ET (0.00 mL): a few voxels decide Dice; WT HD95 74 mm: false-positive region far from the tumor
- **BraTS20_Training_335**: low-grade glioma (under-represented: ~20% of training data); tiny true ET (0.00 mL): a few voxels decide Dice
- **BraTS20_Training_280**: low-grade glioma (under-represented: ~20% of training data); tiny true TC (0.83 mL): a few voxels decide Dice; WT HD95 88 mm: false-positive region far from the tumor
