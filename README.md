# Brain tumor segmentation (BraTS 2020)

Segments gliomas in brain MRI into the three regions clinicians care about —
**whole tumor (WT)**, **tumor core (TC)** and **enhancing tumor (ET)** — from four MRI
sequences (FLAIR, T1, T1ce, T2), and reports their volumes in mL with a per-voxel
uncertainty map. Three 2D U-Net variants are compared, and the models keep working
when a sequence is missing.

**Live demo:** _link added after deployment_ · research / education use only, not for diagnosis.

![App screenshot](docs/app_screenshot.png)

## Why this is hard

Gliomas vary hugely in size, shape and appearance; the enhancing rim can be a few
millimetres thick; tumor is often < 2% of a slice; scans come from 19 institutions
with different scanners. Real-world scans may also lack a sequence.

## Approach

| | |
|---|---|
| Data | BraTS 2020 training set: 369 patients, 4 co-registered 1 mm³ sequences + expert labels |
| Split | by **patient**, 70/15/15 (258/55/56), stratified by tumor grade |
| Input | per-sequence z-score on brain voxels, 192 px crop around the brain → 160×160 axial slices |
| Models | plain U-Net · Attention U-Net · U-Net with ImageNet ResNet34 encoder |
| Training | Dice + BCE, augmentation, **modality dropout** (randomly blank one sequence), early stopping on per-patient validation Dice |
| Evaluation | full 3D reconstruction at native resolution: Dice, HD95 (mm), volume error (mL); test-time augmentation; missing-sequence test; uncertainty vs. error |
| Deployment | Streamlit app, CPU, half-precision weights |

Every choice is explained in [`docs/decisions.md`](docs/decisions.md); the data
audit is in [`notebooks/01_data_audit.ipynb`](notebooks/01_data_audit.ipynb).

## Results (test set, 56 patients)

_Filled in after the final training runs — see [`notebooks/02_results.ipynb`](notebooks/02_results.ipynb)._

| model | Dice WT | Dice TC | Dice ET | HD95 WT (mm) |
|---|---|---|---|---|
| U-Net | – | – | – | – |
| Attention U-Net | – | – | – | – |
| ResNet34 U-Net | – | – | – | – |

## Run it locally

```bash
git clone https://github.com/akhilnathreddy/brain-tumor-seg && cd brain-tumor-seg
python3 -m venv .venv && source .venv/bin/activate

# just the app (uses the weights in models/)
pip install -r requirements.txt
streamlit run app.py

# full pipeline
pip install -r requirements-train.txt
kaggle datasets download awsaf49/brats20-dataset-training-validation -p data/raw   # needs a Kaggle API token
python -m src.preprocess          # ~6 min, ~11 GB cache in data/processed
python -m src.audit               # data audit → reports/audit/
python -m src.train --run configs/runs/resunet34.yaml --estimate   # time estimate
python -m src.train --run configs/runs/resunet34.yaml              # also: unet, attunet, resunet34_nomd
python -m src.evaluate            # test set, once → reports/eval/test/
python -m src.report              # tables + figures
python -m src.export --benchmark  # fp16 weights + ONNX → models/, speed → reports/benchmark.json
```

On Apple Silicon training uses the GPU via MPS automatically
(`PYTORCH_ENABLE_MPS_FALLBACK=1` is set for unsupported ops).

### Train on a free Kaggle GPU instead

The final models were trained this way: one unattended run on a Kaggle T4 that
preprocesses the data, trains all four models, evaluates on the test set and
exports the weights.

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4   # needs a phone-verified Kaggle account
kaggle kernels status <your-username>/brain-tumor-seg-train
kaggle kernels output <your-username>/brain-tumor-seg-train -p kaggle-out   # when COMPLETE
```
Change the `id` in `kaggle/kernel-metadata.json` to your username first.

## Repository

```
src/          data loading, preprocessing, models, training, evaluation, inference, export
configs/      data settings, patient split, one YAML per training run
notebooks/    01 data audit · 02 results
models/       exported half-precision weights used by the app
examples/     3 test-set patients bundled with the app
kaggle/       script + settings to train everything on a Kaggle GPU
reports/      audit and evaluation outputs
docs/         decisions.md (why), app screenshot
app.py        Streamlit demo
```

## Limitations

- **2D, slice by slice.** The model sees one axial slice at a time, so it can't use
  context from neighbouring slices the way a 3D network does; predictions can be
  slightly inconsistent between slices.
- **Pre-operative gliomas only.** No post-surgery scans, metastases or other tumor types.
- **Research data that is already preprocessed.** BraTS scans are skull-stripped,
  co-registered and resampled to 1 mm. Raw hospital scans need the same steps first
  (e.g. HD-BET for skull stripping, registration to a common template).
- **Low-grade gliomas are under-represented** (~20% of patients), and many have no
  enhancing tumor, which makes ET hard to score for them.
- Not validated for clinical use.

## Data and citation

BraTS 2020 (Kaggle copy `awsaf49/brats20-dataset-training-validation`). As requested by the organizers, please cite:

1. B. H. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", *IEEE TMI* 34(10), 2015. doi:10.1109/TMI.2014.2377694
2. S. Bakas et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", *Scientific Data* 4:170117, 2017. doi:10.1038/sdata.2017.117
3. S. Bakas et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv:1811.02629, 2018.
