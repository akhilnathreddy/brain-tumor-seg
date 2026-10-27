"""Export trained models for the app and measure inference speed.

    python -m src.export                 # every checkpoints/*/best.pt
    python -m src.export --benchmark     # also time one full test patient

For each model writes
  models/<name>.pt    half-precision weights (half the size, same predictions)
  models/<name>.onnx  ONNX graph (dynamic batch) for runtimes without PyTorch
and, with --benchmark, reports/benchmark.json: seconds per patient for
PyTorch on this machine's accelerator, PyTorch CPU and ONNX Runtime CPU.
"""
import argparse
import json
import platform
import time

import numpy as np
import torch

from src.brats_io import BraTSSource, check_and_load
from src.inference import OnnxPredictor, TorchPredictor, load_model, predict_volume, to_regions
from src.utils import ROOT, get_device, load_config


def export_half(name):
    ckpt = torch.load(ROOT / "checkpoints" / name / "best.pt", map_location="cpu", weights_only=False)
    out = ROOT / "models" / f"{name}.pt"
    torch.save({"state_dict": {k: v.half() if v.is_floating_point() else v for k, v in ckpt["model"].items()},
                "model_name": ckpt["model_name"], "val_dice": ckpt["val_dice"], "epoch": ckpt["epoch"]}, out)
    return out


def export_onnx(name, size):
    model, _ = load_model(ROOT / "checkpoints" / name / "best.pt")
    out = ROOT / "models" / f"{name}.onnx"
    torch.onnx.export(model, torch.randn(1, 4, size, size), out, input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return out


def check_same(name, size):
    """ONNX and half-precision weights must give (almost) the same output as the checkpoint."""
    ref, _ = load_model(ROOT / "checkpoints" / name / "best.pt")
    half, _ = load_model(ROOT / "models" / f"{name}.pt")
    x = torch.randn(4, 4, size, size)
    with torch.no_grad():
        a, b = torch.sigmoid(ref(x)), torch.sigmoid(half(x))
    c = torch.sigmoid(OnnxPredictor(ROOT / "models" / f"{name}.onnx")(x))
    return float((a - b).abs().max()), float((a - c).abs().max())


def benchmark(name, cfg, pid):
    vol, _, _ = check_and_load(BraTSSource(ROOT / cfg["raw_path"]), pid)
    model, _ = load_model(ROOT / "models" / f"{name}.pt")
    runs = {}
    dev = get_device()
    targets = [(f"pytorch_{dev.type}", TorchPredictor(model.to(dev), dev))]
    if dev.type != "cpu":
        targets.append(("pytorch_cpu", TorchPredictor(load_model(ROOT / "models" / f"{name}.pt")[0], "cpu")))
    targets.append(("onnxruntime_cpu", OnnxPredictor(ROOT / "models" / f"{name}.onnx")))
    masks = {}
    for label, pred in targets:
        predict_volume(pred, vol, cfg)                       # warm-up
        t = time.time()
        res = predict_volume(pred, vol, cfg)
        runs[label] = round(time.time() - t, 2)
        masks[label] = to_regions(res["probs"].astype(np.float32))
    ref = next(iter(masks.values()))
    agree = {k: float((v == ref).mean()) for k, v in masks.items()}
    return {"seconds_per_patient": runs, "voxel_agreement": agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    names = args.models or sorted(p.parent.name for p in (ROOT / "checkpoints").glob("*/best.pt"))
    (ROOT / "models").mkdir(exist_ok=True)
    report = {"machine": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
              "torch": torch.__version__, "models": {}}
    pid = json.loads((ROOT / cfg["splits_file"]).read_text())["test"][0]
    for n in names:
        pt, onnx = export_half(n), export_onnx(n, cfg["img_size"])
        d_half, d_onnx = check_same(n, cfg["img_size"])
        info = {"pt_mb": round(pt.stat().st_size / 1e6, 1), "onnx_mb": round(onnx.stat().st_size / 1e6, 1),
                "max_prob_diff_half": round(d_half, 5), "max_prob_diff_onnx": round(d_onnx, 6)}
        if args.benchmark:
            info.update(benchmark(n, cfg, pid))
        report["models"][n] = info
        print(n, info)
    if args.benchmark:
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports" / "benchmark.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
