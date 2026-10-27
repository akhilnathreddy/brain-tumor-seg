"""Brain tumor segmentation demo (BraTS 2020) — Streamlit, single page.

    streamlit run app.py
"""
import gzip
import hashlib
import json
import urllib.request

import nibabel as nib
import numpy as np
import streamlit as st

from src import viz
from src.inference import TorchPredictor, load_model, load_sequences, predict_volume, regions_to_labels, to_regions
from src.metrics import dice
from src.preprocess import labels_to_regions
from src.utils import REGIONS, ROOT, SEQUENCES, get_device, load_config

st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide")

CFG = load_config()
MODELS_DIR, EXAMPLES_DIR = ROOT / "models", ROOT / "examples"
MODEL_LABELS = {"resunet34": "ResNet34 U-Net", "attunet": "Attention U-Net", "unet": "U-Net",
                "resunet34_nomd": "ResNet34 U-Net (no modality dropout)"}
SEQ_LABELS = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1ce (contrast-enhanced)", "t2": "T2"}
REGION_NAMES = {"WT": "Whole tumor", "TC": "Tumor core", "ET": "Enhancing tumor"}


# ---------------------------------------------------------------- models


def weight_urls():
    """Optional download links for weights that aren't in the repo (models/weights.json)."""
    p = MODELS_DIR / "weights.json"
    return json.loads(p.read_text()) if p.exists() else {}


def available_models():
    names = {p.stem for p in MODELS_DIR.glob("*.pt")} | set(weight_urls())
    order = list(MODEL_LABELS)
    return sorted(names, key=lambda n: order.index(n) if n in order else 99)


@st.cache_resource(show_spinner="Loading model…")
def get_predictor(name):
    path = MODELS_DIR / f"{name}.pt"
    if not path.exists():
        urllib.request.urlretrieve(weight_urls()[name], path)    # first run only
    device = get_device()
    model, ckpt = load_model(path, device)
    return TorchPredictor(model, device), ckpt


# ---------------------------------------------------------------- inputs


def guess_sequence(filename):
    n = filename.lower()
    for seq, keys in [("t1ce", ["t1ce", "t1c", "t1gd", "t1-gd", "t1_gd", "t1post"]),
                      ("flair", ["flair"]), ("t2", ["t2"]), ("t1", ["t1"])]:
        if any(k in n for k in keys):
            return seq
    return None


def example_inputs():
    info = json.loads((EXAMPLES_DIR / "examples.json").read_text())
    label = {e["pid"]: f"Patient {e['pid'][-3:]} · {'high' if e['grade'] == 'HGG' else 'low'}-grade glioma · "
                       f"whole tumor {e['wt_ml']:.0f} mL" for e in info}
    pid = st.selectbox("Example patient (from the held-out test set)", list(label), format_func=label.get)
    files = {s: (EXAMPLES_DIR / pid / f"{pid}_{s}.nii.gz").read_bytes() for s in SEQUENCES}
    seg_path = EXAMPLES_DIR / pid / f"{pid}_seg.nii.gz"
    return files, (seg_path.read_bytes() if seg_path.exists() else None)


def upload_inputs():
    up = st.file_uploader("Upload the patient's MRI sequences (.nii or .nii.gz) — FLAIR, T1, T1ce, T2",
                          type=["nii", "gz"], accept_multiple_files=True)
    if not up:
        st.info("Upload 1–4 NIfTI files. Names containing *flair*, *t1*, *t1ce* or *t2* are recognized "
                "automatically. Scans must be skull-stripped and co-registered, like BraTS data.")
        return None, None
    files = {}
    cols = st.columns(min(len(up), 4))
    for i, f in enumerate(up[:4]):
        guess = guess_sequence(f.name)
        options = SEQUENCES + ["ignore"]
        seq = cols[i % 4].selectbox(f.name, options, index=options.index(guess) if guess else 4,
                                    format_func=lambda s: SEQ_LABELS.get(s, "— ignore —"), key=f"seq_{f.name}")
        if seq != "ignore":
            if seq in files:
                st.error(f"Two files are assigned to {SEQ_LABELS[seq]}.")
                return None, None
            files[seq] = f.getvalue()
    return files, None


# ---------------------------------------------------------------- prediction


def run(files, model_name, tta):
    """Segment once per (input, model, TTA) and keep the result for this session."""
    h = hashlib.sha1()
    for s in SEQUENCES:
        h.update(files.get(s, b"-")[:1_000_000])
        h.update(str(len(files.get(s, b""))).encode())
    key = f"{h.hexdigest()}-{model_name}-{tta}"
    cache = st.session_state.setdefault("results", {})
    if key not in cache:
        vol, affine, missing = load_sequences(files)
        predictor, _ = get_predictor(model_name)
        with st.spinner("Segmenting all slices… (about 20–60 s on a CPU" + (", ×4 with TTA)" if tta else ")")):
            out = predict_volume(predictor, vol, CFG, tta=tta)
        cache.clear()  # keep memory low: one result per session
        cache[key] = {"vol": vol.astype(np.float16), "affine": affine, "missing": missing,
                      "regions": to_regions(out["probs"].astype(np.float32)),
                      "uncertainty": out["uncertainty"], "brain": out["brain"]}
    return cache[key]


def nifti_bytes(labels, affine):
    img = nib.Nifti1Image(labels.astype(np.uint8), affine)
    return gzip.compress(img.to_bytes())


# ---------------------------------------------------------------- page

st.title("Brain tumor segmentation")
st.caption("Segments gliomas in brain MRI into **whole tumor**, **tumor core** and **enhancing tumor**, "
           "using 2D U-Nets trained on BraTS 2020. Pick an example or upload a scan.")

models = available_models()
if not models:
    st.error("No model weights found in models/.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models, format_func=lambda n: MODEL_LABELS.get(n, n))
    tta = st.checkbox("Test-time augmentation", help="Averages predictions over 4 flipped/rotated views: "
                                                     "a little more robust and better uncertainty, 4× slower.")
    st.subheader("Show regions")
    show = (st.checkbox("WT · whole tumor (blue)", True),
            st.checkbox("TC · tumor core (orange)", True),
            st.checkbox("ET · enhancing tumor (green)", True))
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)
    base_seq = st.selectbox("Background sequence", SEQUENCES, format_func=SEQ_LABELS.get)

source = st.radio("Input", ["Example patient", "Upload my own scan"], horizontal=True, label_visibility="collapsed")
files, seg_bytes = example_inputs() if source == "Example patient" else upload_inputs()
if not files:
    st.stop()

missing = [s for s in SEQUENCES if s not in files]
if missing:
    st.warning("Missing sequence(s): **" + ", ".join(SEQ_LABELS[s] for s in missing) + "**. They are filled "
               "with zeros; the models were trained with randomly dropped sequences so they still work, "
               "but expect lower accuracy (especially for the tumor core without T1ce).")

try:
    res = run(files, model_name, tta)
except Exception as e:  # bad upload: wrong shapes, not NIfTI, ...
    st.error(f"Could not process these files: {e}")
    st.stop()

reg, unc, brain, vol = res["regions"], res["uncertainty"], res["brain"], res["vol"]
area = reg[0].sum((0, 1))
tumor_z = np.where(area > 0)[0]

c = st.columns(4)
for i, r in enumerate(REGIONS):
    c[i].metric(f"{REGION_NAMES[r]} ({r})", f"{reg[i].sum() / 1000:.1f} mL")
c[3].metric("Tumor visible in slices", f"{tumor_z.min()}–{tumor_z.max()}" if tumor_z.size else "none")
c[3].caption(f"{tumor_z.size} of {reg.shape[-1]} slices")

z = st.slider("Slice (axial)", 0, reg.shape[-1] - 1, int(area.argmax()) if tumor_z.size else reg.shape[-1] // 2)
bg = vol[SEQUENCES.index(base_seq), :, :, z]
if base_seq in missing:
    bg = vol[SEQUENCES.index([s for s in SEQUENCES if s not in missing][0]), :, :, z]
# show only the brain's bounding box (same box for every slice) so images are larger
xs, ys = np.where(brain.any(2))
box = (slice(max(xs.min() - 4, 0), xs.max() + 5), slice(max(ys.min() - 4, 0), ys.max() + 5)) if xs.size \
    else (slice(None), slice(None))
bg, b2 = bg[box], brain[:, :, z][box]

col1, col2, col3 = st.columns(3)
col1.image(viz.gray(bg, b2), caption=f"Original · {SEQ_LABELS[base_seq]}", use_container_width=True)
col2.image(viz.overlay(bg, reg[:, :, :, z][(slice(None),) + box], show, alpha, b2), caption="Segmentation", use_container_width=True)
col3.image(viz.uncertainty_rgb(unc[:, :, z][box], b2), caption="Uncertainty (white = confident, dark blue = unsure)",
           use_container_width=True)

st.plotly_chart(viz.area_chart(reg, z), use_container_width=True)

if seg_bytes is not None:
    with st.expander("Compare with the expert label (example patients only)"):
        gt = labels_to_regions(np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(seg_bytes)).dataobj))
        g1, g2, g3 = st.columns(3)
        g1.image(viz.overlay(bg, gt[:, :, :, z][(slice(None),) + box], show, alpha, b2), caption="Expert label", use_container_width=True)
        g2.image(viz.overlay(bg, reg[:, :, :, z][(slice(None),) + box], show, alpha, b2), caption="Model", use_container_width=True)
        g3.markdown("**Dice, whole scan (3D)**  \n1 = perfect overlap")
        for i, r in enumerate(REGIONS):
            g3.metric(REGION_NAMES[r], f"{dice(reg[i], gt[i]):.3f}")

with st.expander("3D view"):
    st.plotly_chart(viz.view_3d(reg, brain), use_container_width=True)

st.download_button("Download segmentation (.nii.gz, BraTS labels 1/2/4)",
                   nifti_bytes(regions_to_labels(reg), res["affine"]), file_name="segmentation.nii.gz")

st.divider()
st.caption("⚠️ **Research / education use only — not for diagnosis.** 2D slice-based models trained on "
           "pre-operative glioma scans that are already skull-stripped and co-registered; raw hospital "
           "scans need that preprocessing first.")
st.caption("Data: BraTS 2020 — Menze et al., IEEE TMI 2015; Bakas et al., Sci. Data 2017; "
           "Bakas et al., arXiv:1811.02629, 2018.")
