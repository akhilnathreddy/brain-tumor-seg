# Data audit — BraTS 2020 training set

- Patients found: **369**, usable: **369**, skipped: none
- File issues: BraTS20_Training_355: misnamed seg file 'W39_1998.09.19_Segm.nii' (used it)
- Shapes: {'240x240x155': 369}; voxel size: {'1x1x1': 369}
- Label sets: {'0,1,2,4': 340, '0,1,2': 27, '0,2,4': 1, '0,1,4': 1}
- Image dtypes: {'int16': 335, 'float32': 34}; seg dtypes: {'uint8': 255, 'int16': 71, 'uint16': 43}
- Tumor voxels outside the brain mask: 1248
- Largest brain extent (px): [157, 193]; crop sides used: {192: 358, 195: 4, 193: 3, 194: 1, 197: 1, 201: 1, 196: 1}
- No patient in two splits: **True**

## Splits

| split | patients | HGG | LGG | cached slices | with tumor | tumor-free |
|---|---|---|---|---|---|---|
| train | 258 | 205 | 53 | 34,528 | 16,995 | 17,533 |
| val | 55 | 44 | 11 | 7,481 | 3,734 | 3,747 |
| test | 56 | 44 | 12 | 7,472 | 3,693 | 3,779 |

Training slices per epoch (all tumor + 2,630 sampled tumor-free): **19,625**

## Tumor volume (mL)

| region | mean | median | min | max | patients with 0 |
|---|---|---|---|---|---|
| WT | 99.5 | 90.7 | 7.28 | 361.8 | 0 |
| TC | 41.8 | 33.8 | 0.56 | 190.2 | 0 |
| ET | 19.7 | 15.1 | 0.0 | 111.2 | 27 |

## Resampling ceiling (GT → 160 px → native, Dice)

| region | mean | min |
|---|---|---|
| WT | 0.9853 | 0.9534 |
| TC | 0.9843 | 0.9256 |
| ET | 0.9565 | 0.2222 |

## Disk

- Raw zip: 4.2G
- Processed cache: 11G
