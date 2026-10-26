"""Data audit: summary numbers + figures from the preprocessing outputs.

Run after src.preprocess:  python -m src.audit
Writes reports/audit/summary.json, summary.md and PNG figures.
"""
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.data import BraTSSlices
from src.preprocess import bits_to_regions
from src.utils import REGION_COLORS, REGIONS, ROOT, SEQUENCES, load_config, set_seed

OUT = ROOT / "reports" / "audit"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL = "#b4b2a9"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
})


def du(path):
    return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


def show(ax, img, title=None):
    """Grayscale slice with the (zero) background drawn black."""
    brain = img != 0
    lo, hi = np.percentile(img[brain], [1, 99]) if brain.any() else (0, 1)
    ax.imshow(np.rot90(np.where(brain, img, lo)), cmap="gray", vmin=lo, vmax=hi)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, color=MUTED)


def contour(ax, regions):
    for r, name in enumerate(REGIONS):
        if regions[r].any():
            ax.contour(np.rot90(regions[r]), levels=[0.5], colors=REGION_COLORS[name],
                       linewidths=1.5)


def region_legend(fig):
    handles = [Line2D([], [], color=REGION_COLORS[n], lw=2, label=n) for n in REGIONS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))


def summarize(pat, sl, cfg):
    ok = pat[pat.status == "ok"]
    s = {"patients_found": int(len(pat)), "patients_ok": int(len(ok)),
         "patients_skipped": pat.loc[pat.status != "ok", "pid"].tolist(),
         "issues": {r.pid: r.issues for r in pat.itertuples() if isinstance(r.issues, str) and r.issues}}
    s["confirmed"] = {
        "shapes": ok["shape"].value_counts().to_dict(),
        "voxel_size_mm": ok["zooms"].value_counts().to_dict(),
        "label_sets": ok["labels"].value_counts().to_dict(),
        "image_dtypes": ok["img_dtypes"].value_counts().to_dict(),
        "seg_dtypes": ok["seg_dtype"].value_counts().to_dict(),
        "tumor_voxels_outside_brain_total": int(ok["tumor_outside_brain_vox"].sum()),
        "brain_extent_px_max": [int(ok.brain_dx.max()), int(ok.brain_dy.max())],
        "crop_side_counts": ok["side"].value_counts().to_dict(),
        "brain_volume_ml_mean": round(float(ok.brain_ml.mean()), 1),
    }
    # splits
    splits = json.loads((ROOT / cfg["splits_file"]).read_text())
    sets = [set(v) for v in splits.values()]
    s["no_patient_overlap"] = all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    per = {}
    for k in ["train", "val", "test"]:
        p, d = ok[ok.split == k], sl[sl.split == k]
        n_empty = int((d.WT_px == 0).sum())
        per[k] = {"patients": int(len(p)), "HGG": int((p.grade == "HGG").sum()),
                  "LGG": int((p.grade == "LGG").sum()), "slices_cached": int(len(d)),
                  "tumor_slices": int((d.WT_px > 0).sum()), "tumor_free_slices": n_empty}
    tr = per["train"]
    tr["slices_per_epoch"] = tr["tumor_slices"] + int(round(tr["tumor_free_slices"] * cfg["empty_slice_fraction"]))
    s["splits"] = per
    # volumes
    vol = {}
    for r in REGIONS:
        v = ok[f"vol_{r}_ml"]
        vol[r] = {"mean": round(v.mean(), 1), "median": round(v.median(), 1),
                  "min": round(v.min(), 2), "max": round(v.max(), 1),
                  "patients_with_zero": int((v == 0).sum())}
    s["tumor_volume_ml"] = vol
    s["resampling_ceiling_dice"] = {r: {"mean": round(ok[f"ceiling_dice_{r}"].mean(), 4),
                                        "min": round(ok[f"ceiling_dice_{r}"].min(), 4)} for r in REGIONS}
    s["disk"] = {"raw_zip": du(ROOT / cfg["raw_path"]),
                 "processed_cache": du(ROOT / cfg["processed_dir"])}
    return s


def fig_volumes(ok):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    rng = np.random.default_rng(0)
    for ax, r in zip(axes, REGIONS):
        for i, g in enumerate(["HGG", "LGG"]):
            v = ok.loc[ok.grade == g, f"vol_{r}_ml"].to_numpy()
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False,
                       medianprops={"color": INK}, boxprops={"color": MUTED},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED})
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, alpha=0.55,
                       color=REGION_COLORS[r], linewidths=0)
        ax.set_xticks([0, 1], [f"HGG (n={(ok.grade == 'HGG').sum()})", f"LGG (n={(ok.grade == 'LGG').sum()})"])
        ax.set_title(f"{r} volume", loc="left", fontsize=11)
        ax.set_ylabel("mL")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Tumor volume per patient, by region and grade", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "tumor_volumes.png", dpi=130, bbox_inches="tight")


def fig_slices(summary):
    per = summary["splits"]
    names = ["train", "val", "test"]
    tumor = [per[k]["tumor_slices"] for k in names]
    empty = [per[k]["tumor_free_slices"] for k in names]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(names, tumor, color=REGION_COLORS["WT"], height=0.55, label="with tumor")
    ax.barh(names, empty, left=tumor, color=NEUTRAL, height=0.55, label="tumor-free (brain only)",
            edgecolor=SURFACE, linewidth=2)
    for i, (t, e) in enumerate(zip(tumor, empty)):
        ax.text(t + e + 200, i, f"{t:,} / {e:,}", va="center", color=MUTED, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cached 2D slices")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, (tumor[0] + empty[0]) * 1.22)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Slices with vs without tumor, per split", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "slices_per_split.png", dpi=130, bbox_inches="tight")


def pick_examples(ok):
    """Largest-tumor HGG, an LGG, and the smallest tumor."""
    hgg = ok[ok.grade == "HGG"].sort_values("vol_WT_ml")
    lgg = ok[ok.grade == "LGG"].sort_values("vol_WT_ml")
    return [hgg.iloc[-1].pid, lgg.iloc[len(lgg) // 2].pid, ok.sort_values("vol_WT_ml").iloc[0].pid]


def load_slice(cfg, sl, pid, which="max"):
    d = sl[sl.pid == pid]
    row = d.loc[d.WT_px.idxmax(), "row"] if which == "max" else d.row.iloc[len(d) // 2]
    base = ROOT / cfg["processed_dir"] / "slices"
    img = np.load(base / f"{pid}_img.npy", mmap_mode="r")[row].astype(np.float32)
    msk = bits_to_regions(np.load(base / f"{pid}_msk.npy", mmap_mode="r")[row])
    return img, msk, int(d[d.row == row].z.iloc[0])


def fig_samples(cfg, ok, sl):
    pids = pick_examples(ok)
    fig, axes = plt.subplots(len(pids), 5, figsize=(12, 2.6 * len(pids)))
    for i, pid in enumerate(pids):
        img, msk, z = load_slice(cfg, sl, pid)
        p = ok[ok.pid == pid].iloc[0]
        for c, s in enumerate(SEQUENCES):
            show(axes[i, c], img[c], s.upper() if i == 0 else None)
        show(axes[i, 4], img[0], "FLAIR + labels" if i == 0 else None)
        contour(axes[i, 4], msk)
        axes[i, 0].text(-0.08, 0.5, f"{pid[-3:]} · {p.grade}\nWT {p.vol_WT_ml:.0f} mL\nslice z={z}",
                        transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=9, color=MUTED)
    region_legend(fig)
    fig.suptitle("Normalized, cropped 160×160 inputs: all 4 sequences with expert labels",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0.04, 0.03, 1, 1))
    fig.savefig(OUT / "samples.png", dpi=130, bbox_inches="tight")


def fig_augmentation(cfg):
    set_seed(0)
    ds = BraTSSlices("train", cfg, train=True, modality_dropout=0.0, seed=0)
    tumor_rows = np.where(ds.all.WT_px.to_numpy() > 1500)[0]
    ds.items = ds.all.iloc[[tumor_rows[len(tumor_rows) // 3]]][["pid", "row"]].to_numpy()
    ds_plain = BraTSSlices("train", cfg, train=False)
    ds_plain.items = ds.items
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    x, y = ds_plain[0]
    show(axes[0], x[0].numpy(), "original")
    contour(axes[0], y.numpy())
    for k in range(1, 6):
        x, y = ds[0]
        show(axes[k], x[0].numpy(), f"augmented #{k}")
        contour(axes[k], y.numpy())
    region_legend(fig)
    fig.suptitle("Training augmentation (FLAIR shown): flip, rotate ±15°, scale, shift, per-sequence brightness/contrast",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "augmentation.png", dpi=130, bbox_inches="tight")


def fig_intensity(cfg, ok):
    base = ROOT / cfg["processed_dir"] / "slices"
    pids = ok.pid.sample(20, random_state=0)
    vals = [[] for _ in SEQUENCES]
    for pid in pids:
        img = np.load(base / f"{pid}_img.npy", mmap_mode="r")[::4].astype(np.float32)
        brain = (img != 0).any(1)
        for c in range(4):
            vals[c].append(img[:, c][brain])
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharey=True)
    for c, s in enumerate(SEQUENCES):
        axes[c].hist(np.concatenate(vals[c]), bins=80, range=(-4, 8), color=MUTED)
        axes[c].set_title(s.upper(), loc="left", fontsize=10)
        axes[c].set_xlabel("z-score")
        axes[c].grid(axis="x", visible=False)
    axes[0].set_ylabel("brain voxels")
    fig.suptitle("Intensities after per-sequence normalization (brain voxels, 20 patients)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "intensity.png", dpi=130, bbox_inches="tight")


def to_markdown(s):
    L = ["# Data audit — BraTS 2020 training set", ""]
    L += [f"- Patients found: **{s['patients_found']}**, usable: **{s['patients_ok']}**, "
          f"skipped: {s['patients_skipped'] or 'none'}"]
    L += [f"- File issues: " + ("; ".join(f"{k}: {v}" for k, v in s["issues"].items()) or "none")]
    c = s["confirmed"]
    L += [f"- Shapes: {c['shapes']}; voxel size: {c['voxel_size_mm']}",
          f"- Label sets: {c['label_sets']}",
          f"- Image dtypes: {c['image_dtypes']}; seg dtypes: {c['seg_dtypes']}",
          f"- Tumor voxels outside the brain mask: {c['tumor_voxels_outside_brain_total']}",
          f"- Largest brain extent (px): {c['brain_extent_px_max']}; crop sides used: {c['crop_side_counts']}",
          f"- No patient in two splits: **{s['no_patient_overlap']}**", ""]
    L += ["## Splits", "", "| split | patients | HGG | LGG | cached slices | with tumor | tumor-free |",
          "|---|---|---|---|---|---|---|"]
    for k, v in s["splits"].items():
        L.append(f"| {k} | {v['patients']} | {v['HGG']} | {v['LGG']} | {v['slices_cached']:,} | "
                 f"{v['tumor_slices']:,} | {v['tumor_free_slices']:,} |")
    L += ["", f"Training slices per epoch (all tumor + {s['splits']['train']['slices_per_epoch'] - s['splits']['train']['tumor_slices']:,} "
          f"sampled tumor-free): **{s['splits']['train']['slices_per_epoch']:,}**", ""]
    L += ["## Tumor volume (mL)", "", "| region | mean | median | min | max | patients with 0 |",
          "|---|---|---|---|---|---|"]
    for r, v in s["tumor_volume_ml"].items():
        L.append(f"| {r} | {v['mean']} | {v['median']} | {v['min']} | {v['max']} | {v['patients_with_zero']} |")
    L += ["", "## Resampling ceiling (GT → 160 px → native, Dice)", "", "| region | mean | min |", "|---|---|---|"]
    for r, v in s["resampling_ceiling_dice"].items():
        L.append(f"| {r} | {v['mean']} | {v['min']} |")
    L += ["", f"## Disk", "", f"- Raw zip: {s['disk']['raw_zip']}", f"- Processed cache: {s['disk']['processed_cache']}"]
    return "\n".join(L) + "\n"


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    pat = pd.read_csv(ROOT / cfg["processed_dir"] / "patients.csv")
    sl = pd.read_csv(ROOT / cfg["processed_dir"] / "slices.csv")
    ok = pat[pat.status == "ok"]
    s = summarize(pat, sl, cfg)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (OUT / "summary.md").write_text(to_markdown(s))
    fig_volumes(ok)
    fig_slices(s)
    fig_samples(cfg, ok, sl)
    fig_augmentation(cfg)
    fig_intensity(cfg, ok)
    print(to_markdown(s))


if __name__ == "__main__":
    main()
