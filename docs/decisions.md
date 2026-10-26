# Key decisions

Short notes on *why* things are done the way they are. Updated as the project goes.

## Data

**Training set only.** The BraTS 2020 validation set ships without labels, so it
cannot be used to measure anything. All train/val/test splits come from the 369
labelled training patients.

**Split by patient, 70/15/15, stratified by grade.** Neighbouring slices of the same
brain are almost identical, so a slice-level split would leak test patients into
training and inflate scores. Stratifying by HGG/LGG keeps the ~20% low-grade share
equal across splits (LGG tumors look different and often have no enhancing part).
The split is saved in `configs/splits.json` and committed, so every run uses
exactly the same patients.

**Verified, not assumed.** Every patient is checked on load: all 5 files present,
identical 240×240×155 shape and affine across sequences (co-registered), 1 mm
voxels, labels ⊆ {0,1,2,4}, finite values. One real issue was found: patient 355's
label file is named `W39_1998.09.19_Segm.nii`; the loader detects and uses it.
File dtypes vary (int16/float32 images; uint8/int16/uint16 labels), so everything is
cast explicitly.

**Read straight from the zip.** The uncompressed training set is ~30 GB; reading
NIfTI bytes from the 4.2 GB zip avoids extracting it. The loader also accepts an
extracted folder.

**Nested regions WT/TC/ET instead of raw labels.** The BraTS challenge evaluates
whole tumor (1+2+4), tumor core (1+4) and enhancing tumor (4), and these are what
clinicians care about (extent, resectable core, active tumor). Predicting them as 3
independent sigmoid channels lets regions overlap as they should.

**Normalization: per-sequence z-score on brain voxels only, after clipping to the
0.5–99.5 percentiles.** MRI intensities have no absolute units and vary between
scanners, so each sequence of each patient is standardized on its own. Using only
brain voxels stops the large zero background from dominating the statistics;
clipping removes a few extreme voxels that would otherwise squash the range.
Background is kept at exactly 0. A missing sequence stays all-zero, which is the
same thing modality dropout shows the model during training.

**Crop 192×192 around the brain, resize to 160×160.** A fixed 192 box fits 358 of
369 brains, so almost every patient has the *same physical scale* (1.2 mm/px after
resizing) and tumor size in pixels means the same thing everywhere. The 11 largest
brains (up to 157×193 px) get a slightly bigger box (≤201) instead of being clipped. 160 rather than 128 keeps small enhancing-tumor
rims: resizing ground truth to 160 px and back still gives Dice ≈ 0.96–0.99 (the
"resampling ceiling" in the audit), i.e. the resolution costs very little.

**Masks resized as soft maps, then thresholded at 0.5.** Bilinear weights are
non-negative, so ET ⊆ TC ⊆ WT survives resizing. Masks are stored as a 1-byte
bitmask per pixel.

**2D axial slices.** A 2D model trains on a 16 GB Mac in reasonable time, while a
3D U-Net on full volumes needs far more memory. The cost is lost through-plane
context (see limitations in the README).

**Cache all brain slices, sample tumor-free ones per epoch.** Slices with almost no
brain (< 300 px and no tumor) are dropped at caching time. During training every
tumor slice is used plus a fresh random 15% of tumor-free brain slices each epoch:
training stays fast and focused on tumors, the model still sees healthy tissue (so
it learns not to hallucinate tumors), and different empty slices are seen over
epochs. Validation uses every cached slice, so it is not biased toward tumors.

**Cache format: one float16 `.npy` per patient, memory-mapped.** Training never
touches NIfTI again; float16 halves disk use with no meaningful precision loss for
z-scored data.

**Augmentation.** albumentations does the geometric part on image + mask:
horizontal flip (left/right hemispheres are interchangeable), rotation ±15°, scale
0.9–1.1, small shifts. No vertical flips: an upside-down brain never occurs.
Brightness/contrast jitter (mimics scanner differences) is applied per sequence as
`x*a + b` inside the brain, written by hand: albumentations'
`RandomBrightnessContrast` clips float images to [0, 1], which on z-scored MRI would
wipe out every negative value (caught in the audit's augmentation figure).

**Label quirks left as they are.** 36 patients have a few labelled tumor voxels
where all four sequences are exactly 0 (worst case 588 voxels, < 0.5% of that
tumor). 27 patients (mostly LGG) have no enhancing tumor at all, so ET Dice for
them depends on predicting *nothing*; this is handled in evaluation (empty
prediction on empty ground truth = Dice 1). Patient 307 has only 32 ET voxels; at
160 px such a tiny region can't be represented exactly (resampling ceiling 0.22).
