"""Train one model.

    python -m src.train --run configs/runs/resunet34.yaml --estimate   # time estimate only
    python -m src.train --run configs/runs/resunet34.yaml              # train
    python -m src.train --run configs/runs/resunet34.yaml --resume     # continue after a stop

Early stopping on mean validation Dice (WT, TC, ET, per patient). Writes
checkpoints/<name>/{best.pt,last.pt,history.csv} and TensorBoard logs in runs/<name>.
"""
import argparse
import csv
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import BraTSSlices, worker_init_fn
from src.losses import DiceBCELoss, PatientDice
from src.models import build_model, count_params
from src.utils import REGIONS, ROOT, get_device, load_config, set_seed


def load_run_config(run_path):
    cfg = load_config("configs/train_base.yaml")
    cfg.update(load_config(run_path))
    return cfg


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def make_loaders(cfg, dcfg, device):
    train_ds = BraTSSlices("train", dcfg, train=True, modality_dropout=cfg["modality_dropout"],
                           seed=cfg["seed"])
    val_ds = BraTSSlices("val", dcfg, return_pid=True)
    kw = dict(num_workers=cfg["num_workers"], worker_init_fn=worker_init_fn,
              pin_memory=device.type == "cuda", persistent_workers=False)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                          generator=g, **kw)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False, **kw)
    return train_ds, train_dl, val_dl


def train_one_epoch(model, dl, opt, loss_fn, device, max_batches=None):
    model.train()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(tqdm(dl, desc="train", leave=False)):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / max(n, 1)


@torch.no_grad()
def validate(model, dl, loss_fn, device, max_batches=None):
    model.eval()
    dice, total, n = PatientDice(), 0.0, 0
    for i, (x, y, pids) in enumerate(tqdm(dl, desc="val", leave=False)):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        n += len(x)
        dice.update(logits, y, pids)
    return total / max(n, 1), dice.mean()


def estimate(model, train_dl, val_dl, opt, loss_fn, device, cfg, n=20, warmup=3):
    """Time a few batches and extrapolate to one epoch and to the full run."""
    def timed(fn, dl, k):
        it = iter(dl)
        for _ in range(warmup):
            fn(next(it))
        sync(device)
        t = time.time()
        for _ in range(k):
            fn(next(it))
        sync(device)
        return (time.time() - t) / k

    def train_step(batch):
        model.train()
        x, y = batch[0].to(device), batch[1].to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    @torch.no_grad()
    def val_step(batch):
        model.eval()
        model(batch[0].to(device))

    t_train = timed(train_step, train_dl, n)
    t_val = timed(val_step, val_dl, n // 2)
    epoch = len(train_dl) * t_train + len(val_dl) * t_val
    full = epoch * cfg["epochs"]
    print(f"\n[{cfg['name']}] on {device}: {t_train * 1000:.0f} ms/train batch, {t_val * 1000:.0f} ms/val batch")
    print(f"  one epoch ≈ {epoch / 60:.1f} min ({len(train_dl)} train + {len(val_dl)} val batches)")
    print(f"  max {cfg['epochs']} epochs ≈ {full / 3600:.1f} h; early stopping typically ends at "
          f"50–70% → ≈ {0.5 * full / 3600:.1f}–{0.7 * full / 3600:.1f} h")
    if full > 2 * 3600:
        print("  NOTE: the worst case is over 2 h.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="configs/runs/<name>.yaml")
    ap.add_argument("--estimate", action="store_true", help="only print a training-time estimate")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>/last.pt")
    ap.add_argument("--epochs", type=int, help="override max epochs")
    ap.add_argument("--max-batches", type=int, help="limit batches per epoch (smoke tests)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline)")
    ap.add_argument("--device", help="force cpu / mps / cuda")
    args = ap.parse_args()

    cfg = load_run_config(args.run)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.no_pretrained:
        cfg["pretrained"] = False
    dcfg = load_config()
    set_seed(cfg["seed"])
    device = torch.device(args.device) if args.device else get_device()

    model = build_model(cfg["model"], pretrained=cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=cfg["lr_patience"])
    loss_fn = DiceBCELoss()
    train_ds, train_dl, val_dl = make_loaders(cfg, dcfg, device)
    print(f"{cfg['name']}: {cfg['model']}, {count_params(model) / 1e6:.1f}M params, device {device}, "
          f"{len(train_ds)} train slices/epoch, {len(val_dl.dataset)} val slices")

    if args.estimate:
        estimate(model, train_dl, val_dl, opt, loss_fn, device, cfg)
        return

    ckpt_dir = ROOT / "checkpoints" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start, best, bad = 1, -1.0, 0
    if args.resume and (ckpt_dir / "last.pt").exists():
        state = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start, best, bad = state["epoch"] + 1, state["best"], state["bad"]
        print(f"resumed at epoch {start}, best val Dice {best:.4f}")
    train_ds.rng = np.random.default_rng(cfg["seed"] + start)  # so a resume samples new slices

    writer = SummaryWriter(ROOT / "runs" / cfg["name"])
    hist_path = ckpt_dir / "history.csv"
    new_hist = not (args.resume and hist_path.exists())
    with open(hist_path, "w" if new_hist else "a", newline="") as f:
        if new_hist:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", *[f"dice_{r}" for r in REGIONS],
                                    "dice_mean", "lr", "minutes"])

    for epoch in range(start, cfg["epochs"] + 1):
        t0 = time.time()
        train_ds.resample()                       # fresh tumor-free slices each epoch
        tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device, args.max_batches)
        va_loss, dice = validate(model, val_dl, loss_fn, device, args.max_batches)
        score = dice.mean().item()
        sched.step(score)
        lr = opt.param_groups[0]["lr"]
        minutes = (time.time() - t0) / 60

        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("val_dice", {r: dice[i].item() for i, r in enumerate(REGIONS)}, epoch)
        writer.add_scalar("val_dice/mean", score, epoch)
        writer.add_scalar("lr", lr, epoch)
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{va_loss:.4f}",
                                    *[f"{d:.4f}" for d in dice.tolist()], f"{score:.4f}", lr, f"{minutes:.1f}"])

        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "model_name": cfg["model"], "config": cfg,
                        "epoch": epoch, "val_dice": dict(zip(REGIONS, dice.tolist()))},
                       ckpt_dir / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "best": best, "bad": bad},
                   ckpt_dir / "last.pt")
        print(f"epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f} | Dice "
              + " ".join(f"{r} {d:.3f}" for r, d in zip(REGIONS, dice.tolist()))
              + f" | mean {score:.4f}{' *' if improved else ''} | lr {lr:.1e} | {minutes:.1f} min")
        if bad >= cfg["patience"]:
            print(f"early stop: no improvement for {cfg['patience']} epochs")
            break

    writer.close()
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"best mean val Dice {best:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
