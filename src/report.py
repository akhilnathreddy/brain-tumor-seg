"""Tables and figures from src.evaluate output.

    python -m src.report --split test

Writes reports/eval/<split>/results.md plus PNG figures used by
notebooks/02_results.ipynb and the README.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from src.audit import INK, MUTED, NEUTRAL, SURFACE, contour, show  # shared figure style
from src.brats_io import BraTSSource, check_and_load
from src.inference import TorchPredictor, load_model, predict_volume, to_regions
from src.preprocess import labels_to_regions
from src.utils import REGION_COLORS, REGIONS, ROOT, SEQUENCES, get_device, load_config

LABELS = {"unet": "U-Net", "attunet": "Attention U-Net", "resunet34": "ResNet34 U-Net",
          "resunet34_nomd": "ResNet34 U-Net (no modality dropout)"}


def brain_box(vol, pad=4):
    """In-plane bounding box of the brain, so figures aren't mostly black."""
    xs, ys = np.where((vol > 0).any(0).any(-1))
    return (slice(max(xs.min() - pad, 0), xs.max() + pad + 1), slice(max(ys.min() - pad, 0), ys.max() + pad + 1))


def best_model_by_val():
    """Pick the model with the highest validation Dice (never the test score).

    Reads the training checkpoints, or the exported weights in models/ (which keep
    the validation Dice) when the checkpoints aren't on this machine.
    """
    paths = {p.parent.name: p for p in (ROOT / "checkpoints").glob("*/best.pt")} or \
            {p.stem: p for p in (ROOT / "models").glob("*.pt")}
    scores = {name: float(np.mean(list(torch.load(p, map_location="cpu", weights_only=False)["val_dice"].values())))
              for name, p in paths.items()}
    return max(scores, key=scores.get), scores


def fmt(mean, median, digits=3):
    return f"{mean:.{digits}f} ({median:.{digits}f})"


def table_main(df):
    d = df[~df.tta & df["drop"].isna()]
    lines = ["| model | " + " | ".join(f"Dice {r}" for r in REGIONS) + " | "
             + " | ".join(f"HD95 {r} (mm)" for r in REGIONS) + " | "
             + " | ".join(f"\\|vol err\\| {r} (mL)" for r in REGIONS) + " |",
             "|---" * 10 + "|"]
    for m, g in d.groupby("model"):
        cells = [fmt(g[f"dice_{r}"].mean(), g[f"dice_{r}"].median()) for r in REGIONS]
        cells += [fmt(g[f"hd95_{r}"].mean(), g[f"hd95_{r}"].median(), 1) for r in REGIONS]
        cells += [fmt(g[f"vol_err_{r}"].abs().mean(), g[f"vol_err_{r}"].abs().median(), 1) for r in REGIONS]
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_tta(df):
    d = df[df["drop"].isna()]
    lines = ["| model | " + " | ".join(f"Dice {r}: plain → TTA" for r in REGIONS)
             + " | seconds/patient: plain → TTA |", "|---" * 5 + "|"]
    for m, g in d.groupby("model"):
        if g.tta.nunique() < 2:
            continue
        a, b = g[~g.tta], g[g.tta]
        cells = [f"{a[f'dice_{r}'].mean():.3f} → {b[f'dice_{r}'].mean():.3f} "
                 f"({b[f'dice_{r}'].mean() - a[f'dice_{r}'].mean():+.3f})" for r in REGIONS]
        cells.append(f"{a.seconds.mean():.1f} → {b.seconds.mean():.1f}")
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def missing_frame(df):
    d = df[~df.tta & df.model.isin(["resunet34", "resunet34_nomd"])].copy()
    d["removed"] = d["drop"].fillna("none")
    d["dice_mean"] = d[[f"dice_{r}" for r in REGIONS]].mean(1)
    return d


def table_missing(df):
    d = missing_frame(df)
    if d["drop"].isna().all():
        return "_missing-sequence test not run_"
    lines = ["| removed sequence | " + " | ".join(
        f"{LABELS[m]}: " + " / ".join(REGIONS) for m in ["resunet34", "resunet34_nomd"] if m in set(d.model)) + " |",
        "|---" * (1 + d.model.nunique()) + "|"]
    for s in ["none"] + SEQUENCES:
        cells = []
        for m in ["resunet34", "resunet34_nomd"]:
            g = d[(d.model == m) & (d.removed == s)]
            if len(g):
                cells.append(" / ".join(f"{g[f'dice_{r}'].mean():.3f}" for r in REGIONS))
        lines.append(f"| {s.upper() if s != 'none' else '— (all 4)'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_uncertainty(df):
    d = df[df["drop"].isna()]
    lines = ["| model | TTA | AUROC (uncertainty → error) | error rate, 1% most uncertain voxels | error rate, other voxels |",
             "|---|---|---|---|---|"]
    for (m, t), g in d.groupby(["model", "tta"]):
        lines.append(f"| {LABELS.get(m, m)} | {'yes' if t else 'no'} | {g.unc_auroc_WT.mean():.3f} | "
                     f"{g.err_top1pct_WT.mean():.1%} | {g.err_rest_WT.mean():.2%} |")
    return "\n".join(lines)


def failure_notes(r):
    """Plain-language reasons a patient scored badly, from its metrics."""
    notes = []
    if r.grade == "LGG":
        notes.append("low-grade glioma (under-represented: ~20% of training data)")
    for reg in REGIONS:
        gt, pred = r[f"vol_gt_{reg}"], r[f"vol_pred_{reg}"]
        if gt == 0 and pred > 0:
            notes.append(f"no {reg} in the expert label but {pred:.1f} mL predicted (Dice 0 by definition)")
        elif 0 < gt < 1 and r[f"dice_{reg}"] < 0.5:
            notes.append(f"tiny true {reg} ({gt:.2f} mL): a few voxels decide Dice")
    if r.hd95_WT > 30:
        notes.append(f"WT HD95 {r.hd95_WT:.0f} mm: false-positive region far from the tumor")
    ratio = r.vol_pred_WT / max(r.vol_gt_WT, 1e-3)
    if ratio > 1.5:
        notes.append(f"over-segmentation: predicted WT is {ratio:.1f}× the true volume")
    elif ratio < 0.67:
        notes.append(f"under-segmentation: predicted WT is {ratio:.1f}× the true volume")
    return notes or ["no single obvious cause; boundary disagreement"]


def fig_gallery(df, model, split, cfg, out, k=3):
    d = df[(df.model == model) & ~df.tta & df["drop"].isna()].copy()
    d["dice_mean"] = d[[f"dice_{r}" for r in REGIONS]].mean(1)
    d = d.sort_values("dice_mean")
    picks = [("best", r) for r in d.tail(k).iloc[::-1].itertuples()] + \
            [("worst", r) for r in d.head(k).itertuples()]
    device = get_device()
    pred = TorchPredictor(load_model(ROOT / "checkpoints" / model / "best.pt", device)[0], device)
    source = BraTSSource(ROOT / cfg["raw_path"])
    fig, axes = plt.subplots(len(picks), 3, figsize=(10.5, 3.2 * len(picks)))
    for i, (kind, r) in enumerate(picks):
        vol, seg, _ = check_and_load(source, r.pid)
        gt = labels_to_regions(seg)
        res = predict_volume(pred, vol, cfg)
        reg = to_regions(res["probs"].astype(np.float32))
        z = int(np.argmax(gt[0].sum((0, 1)))) if gt[0].any() else vol.shape[-1] // 2
        bx = brain_box(vol)
        flair = vol[0, :, :, z][bx]
        show(axes[i, 0], flair, "FLAIR" if i == 0 else None)
        show(axes[i, 1], flair, "expert label" if i == 0 else None)
        contour(axes[i, 1], gt[:, :, :, z][(slice(None),) + bx])
        show(axes[i, 2], flair, "prediction" if i == 0 else None)
        contour(axes[i, 2], reg[:, :, :, z][(slice(None),) + bx])
        axes[i, 0].text(-0.06, 0.5, f"{kind.upper()}\n{r.pid[-3:]} · {r.grade}\nDice "
                        + " / ".join(f"{getattr(r, f'dice_{x}'):.2f}" for x in REGIONS) + f"\nz={z}",
                        transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=9, color=MUTED)
    handles = [Line2D([], [], color=REGION_COLORS[n], lw=2, label=n) for n in REGIONS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{LABELS.get(model, model)} on the {split} set: best and worst patients (Dice WT / TC / ET)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0.08, 0.02, 1, 1))
    fig.savefig(out / "gallery.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_uncertainty(df, model, split, cfg, out):
    """One median patient: prediction errors next to the confidence map."""
    d = df[(df.model == model) & df.tta & df["drop"].isna()]
    if d.empty:
        d = df[(df.model == model) & ~df.tta & df["drop"].isna()]
    r = d.iloc[(d.dice_WT - d.dice_WT.median()).abs().argsort().iloc[0]]
    device = get_device()
    pred = TorchPredictor(load_model(ROOT / "checkpoints" / model / "best.pt", device)[0], device)
    vol, seg, _ = check_and_load(BraTSSource(ROOT / cfg["raw_path"]), r.pid)
    gt = labels_to_regions(seg)
    res = predict_volume(pred, vol, cfg, tta=bool(r.tta))
    wt = to_regions(res["probs"].astype(np.float32))[0]
    z = int(np.argmax(gt[0].sum((0, 1))))
    bx = brain_box(vol)
    flair, p, g = vol[0, :, :, z][bx], wt[:, :, z][bx], gt[0][:, :, z][bx]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    show(axes[0], flair, "FLAIR + prediction (WT)")
    contour(axes[0], np.stack([p, np.zeros_like(p), np.zeros_like(p)]))
    colors = {"correct": ("#2a78d6", 0.35), "false positive": ("#e34948", 1.0), "false negative": ("#eda100", 1.0)}
    err = np.zeros(p.shape + (4,))
    for name, m in [("correct", p & g), ("false positive", p & ~g), ("false negative", ~p & g)]:
        err[m] = matplotlib.colors.to_rgba(colors[name][0], colors[name][1])
    show(axes[1], flair, "prediction vs expert label")
    axes[1].imshow(np.rot90(err))
    axes[1].legend(handles=[Line2D([], [], color=c, alpha=a, lw=6, label=n) for n, (c, a) in colors.items()],
                   loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=8)
    u = res["uncertainty"][:, :, z][bx].astype(np.float32)
    im = axes[2].imshow(np.rot90(u), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_axis_off()
    axes[2].set_title("uncertainty (0 = sure, 1 = coin flip)", fontsize=10, color=MUTED)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"Uncertain pixels sit where the errors are ({r.pid[-3:]}, WT Dice {r.dice_WT:.2f}, "
                 f"AUROC {r.unc_auroc_WT:.2f})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "uncertainty.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_missing(df, out):
    d = missing_frame(df)
    if d["drop"].isna().all():
        return
    order = ["none"] + SEQUENCES
    models = [m for m in ["resunet34", "resunet34_nomd"] if m in set(d.model)]
    colors = {"resunet34": REGION_COLORS["WT"], "resunet34_nomd": NEUTRAL}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.38
    for j, m in enumerate(models):
        vals = [d[(d.model == m) & (d.removed == s)].dice_mean.mean() for s in order]
        xs = np.arange(len(order)) + (j - (len(models) - 1) / 2) * w
        ax.bar(xs, vals, width=w - 0.03, color=colors[m], label=LABELS[m])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(order)), ["all 4"] + [f"no {s.upper()}" for s in SEQUENCES])
    ax.set_ylabel("mean Dice (WT, TC, ET)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Missing-sequence test: Dice when one sequence is removed", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "missing_sequences.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_models(df, out):
    d = df[~df.tta & df["drop"].isna()]
    models = sorted(d.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, r in zip(axes, REGIONS):
        for i, m in enumerate(models):
            v = d.loc[d.model == m, f"dice_{r}"].to_numpy()
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops={"color": INK},
                       boxprops={"color": MUTED}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, alpha=0.5,
                       color=REGION_COLORS[r], linewidths=0)
        ax.set_xticks(range(len(models)), [LABELS.get(m, m).replace(" (no modality dropout)", "\n(no MD)")
                                           .replace(" U-Net", "\nU-Net") for m in models], fontsize=8)
        ax.set_title(f"{r} Dice per patient", loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--no-images", action="store_true", help="tables only (no re-prediction)")
    args = ap.parse_args()
    cfg = load_config()
    out = ROOT / "reports" / "eval" / args.split
    df = pd.read_csv(out / "per_patient.csv")
    df["tta"] = df.tta.astype(bool)
    best, val_scores = best_model_by_val()

    md = [f"# Results — {args.split} set ({df.pid.nunique()} patients)", "",
          f"Best model by validation Dice: **{LABELS.get(best, best)}** "
          f"(val mean Dice {val_scores[best]:.3f}). Values are mean (median) over patients.", "",
          "## Model comparison (no TTA)", "", table_main(df), "",
          "## Test-time augmentation", "", table_tta(df), "",
          "## Missing-sequence test (mean Dice WT / TC / ET)", "", table_missing(df), "",
          "## Does uncertainty point at errors? (WT)", "", table_uncertainty(df), ""]
    fig_models(df, out)
    fig_missing(df, out)
    if not args.no_images:
        fig_gallery(df, best, args.split, cfg, out)
        fig_uncertainty(df, best, args.split, cfg, out)
    d = df[(df.model == best) & ~df.tta & df["drop"].isna()].copy()
    d["dice_mean"] = d[[f"dice_{r}" for r in REGIONS]].mean(1)
    md += ["## Why the worst patients fail", ""]
    md += [f"- **{r.pid}** (Dice " + " / ".join(f"{r[f'dice_{x}']:.2f}" for x in REGIONS) + "): "
           + "; ".join(failure_notes(r)) for _, r in d.nsmallest(3, "dice_mean").iterrows()]
    (out / "results.md").write_text("\n".join(md) + "\n")
    (out / "best_model.json").write_text(json.dumps({"best": best, "val_mean_dice": val_scores}, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
