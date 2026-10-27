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

## Models and training

**Three architectures, same everything else.** All models get identical data,
augmentation, loss, optimizer schedule and early stopping, so differences in the
results come from the architecture alone.
- *Plain U-Net* (7.8M params, from scratch): the standard baseline for medical
  segmentation; skip connections carry fine detail from encoder to decoder.
- *Attention U-Net* (7.9M): the same U-Net plus attention gates on each skip
  connection. The gate uses the coarser decoder signal to down-weight skip features
  away from the tumor, which helps with small targets like enhancing tumor. Costs
  only ~0.1M extra parameters.
- *ResNet34 U-Net* (24.4M, `segmentation_models_pytorch`): encoder pretrained on
  ImageNet. Natural-image edges and textures transfer surprisingly well to MRI and
  usually speed up convergence. smp adapts the first conv to 4 input channels by
  reusing the pretrained RGB filters (cycled and rescaled), so no pretrained
  weights are thrown away. It gets a lower learning rate (3e-4 vs 1e-3) so the
  pretrained features aren't destroyed in the first epochs.

**3 sigmoid outputs, not a 4-class softmax.** WT ⊃ TC ⊃ ET overlap, so each region
is its own yes/no question. This also optimizes directly for the regions BraTS
evaluates.

**Loss: Dice + BCE.** BCE gives stable, well-behaved gradients per pixel; Dice
optimizes overlap directly and is insensitive to the huge background/tumor
imbalance (tumor is often < 2% of a slice). Soft Dice is computed over the whole
batch per region, so batches with no tumor don't produce 0/0.

**Modality dropout (p = 0.3).** During training, 30% of slices have one random
sequence set to zero, which is exactly what a missing sequence looks like after
our preprocessing. The model learns not to depend on any single sequence. To
measure whether this actually helps, the ResNet34 model is trained twice: with
(`resunet34`) and without (`resunet34_nomd`) modality dropout.

**Early stopping on per-patient validation Dice.** Validation Dice is accumulated
per patient over all of that patient's slices and averaged over patients and the
three regions. Per patient (not per slice) because that is how the test set is
scored and how a clinician sees results: one score per scan. Empty prediction on
empty ground truth counts as 1, otherwise the 27 patients without ET would be
scored 0 no matter what. Training stops after 6 epochs without improvement (max
30); the best checkpoint is kept. The LR is halved after 3 flat epochs.

**AdamW, batch 16, fp32.** AdamW is a robust default; batch 16 at 160×160 fits
easily in 16 GB unified memory. Mixed precision is left off for MPS stability
(MPS autocast support is still uneven).

**Reproducibility.** Seed 42 for Python/NumPy/torch, a seeded DataLoader
generator, and per-worker augmentation seeds derived from it (without this,
parallel workers would produce identical augmentations).

**Resume support.** A Mac that sleeps or a run that's interrupted can continue
from `last.pt` with `--resume` instead of starting over.

**Where the final training ran: a free Kaggle T4.** On the M2 Pro, the ResNet34 U-Net
alone was estimated at 0.8–1.1 h (3.1 min/epoch), but all four runs add up to
4–6 h of a busy laptop. One unattended Kaggle session (`kaggle/train_kaggle.py`)
runs the identical code end to end: it clones this repo, reads the dataset as an
extracted folder (the loader handles both zip and folder), trains the four
models, evaluates on the test set and exports the weights. The patient split comes
from `configs/splits.json`, so the data is split exactly as in local runs.

**No open files per patient.** The first Mac run failed with "Too many open
files": memory-mapping the cache kept 2 files per patient open in every
DataLoader worker (~500), above macOS's default limit of 256. Each slice is now
read with a plain seek + read (the file header is parsed once and its offset
remembered), which is just as fast and keeps no files open.

## Evaluation

**Scored in 3D, at native resolution.** Slice predictions are stacked, resized back
from 160 px to the crop size and pasted into the original 240×240×155 grid, so
Dice, HD95 and volumes are measured exactly where the expert labels live (1 mm³
voxels: 1000 voxels = 1 mL). Scoring at 160 px would flatter the model.

**Test set used once.** Model choice uses validation Dice only (`best.pt` stores
it); `src.evaluate` defaults to the test split and is meant to be run a single time
at the end. `--split val` exists for development.

**Metrics.** Dice measures overlap; HD95 (95th-percentile surface distance, mm)
catches what Dice hides — a small false-positive blob far from the tumor barely
changes Dice but ruins HD95; volume error in mL is what a clinician would read off
a report. Mean *and* median are reported because a few failures drag the mean
down. Conventions (as in BraTS): empty prediction on empty truth → Dice 1, HD95 0;
exactly one empty → HD95 373 mm (the image diagonal).

**Post-processing kept minimal.** Probabilities are thresholded at 0.5 and nesting
is enforced (ET ⊆ TC ⊆ WT) so the output is anatomically consistent; nothing
outside the brain mask is kept. No connected-component tricks, so the numbers show
what the network itself does.

**Test-time augmentation.** Average of 4 views — original, horizontal flip, ±10°
rotation — all inside the training augmentation range (views the model never saw,
like vertical flips, would hurt). Predictions are mapped back before averaging.
Costs 4× time; the report shows whether it pays off.

**Uncertainty = entropy of the averaged probability.** For each voxel, binary
entropy of the mean probability, maximum over the 3 regions: 0 when the model is
sure, 1 at p = 0.5. It works for every model, with or without TTA (with TTA the
average also absorbs disagreement between views, which is the TTA uncertainty
the brief asked for). MC dropout was not used: the pretrained ResNet34 U-Net has
no dropout layers, and one method for all models keeps the comparison fair. The
check that it is meaningful: AUROC of uncertainty for predicting wrong voxels,
and the error rate among the 1% most uncertain voxels vs. the rest.

**Missing-sequence test.** Each sequence is zeroed in turn (exactly what a missing
file becomes in the app) for the ResNet34 U-Net trained with and without modality
dropout. Same architecture, same seed: any difference is due to modality dropout.

## Export and app

**Half-precision weights for the app.** fp16 halves the file (ResNet34 U-Net:
98 → 49 MB, under GitHub's 50 MB warning) with a maximum probability change of
~1e-4; weights are cast back to fp32 on load because CPU fp16 inference is slow.

**ONNX for speed measurements.** ONNX Runtime ran ~2× faster than PyTorch on CPU
in a first test. The app still uses PyTorch: the fp32 ONNX file (98 MB) is close
to GitHub's 100 MB hard limit, and keeping one code path means the app runs exactly
the evaluated pipeline.

**App design.** One page, no login or database. A prediction is computed once per
(input, model, TTA) and kept in the session; only one result is kept to stay inside
the free tier's memory. Views are cropped to the brain so images are larger.
Missing sequences are allowed (zero-filled, with a warning) because the models
were trained for it. Device is auto-detected (CPU on Streamlit Cloud).

**Bundled examples.** Three test-set patients (never used in training or model
selection): a large high-grade tumor, a low-grade tumor and the smallest tumor,
~8 MB each as compressed NIfTI with their expert labels, so visitors can compare.
Licensing: the Kaggle copy is marked CC0; the BraTS organizers' terms allow use
provided their papers are cited and say nothing specific about redistribution.
Bundling three cases for an educational demo with full citations (app footer and
README) is in the spirit of those terms.
