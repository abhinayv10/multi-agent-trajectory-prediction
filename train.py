"""
train.py

Purpose:
    Train a trajectory model (full model, ablation or LSTM baseline).
    All hyperparameters come from a YAML config; any key can be overridden
    from the command line, so experiments never need edited config copies.

    The checkpoint kept is the epoch with the best validation
    minADE@K + minFDE@K (K = model.num_predictions); early stopping watches
    the same number. Validation latents are drawn with a fixed seed, so
    epochs are compared on identical samples.

Usage:
    python train.py --config configs/eth_ucy.yaml
    python train.py --config configs/eth_ucy.yaml \
        --set dataset.test_scene=hotel model.use_social=false --seed 1 \
        --checkpoint runs/eth_ucy/hotel/no_social/seed1/best.pt
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import DatasetFactory, build_dataloader
from models import build_model
from utils.config import load_config
from utils.losses import compute_total_loss
from utils.metrics import per_agent_errors, summarize_errors


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed all RNGs (Python, NumPy, torch CPU + CUDA) for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_import_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# LR warmup wrapper
# ---------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs`, then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.cosine        = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=base_lr * 0.01,
        )
        self._epoch = 0
        if warmup_epochs > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr / warmup_epochs

    def step(self):
        self._epoch += 1
        if self._epoch < self.warmup_epochs:
            lr = self.base_lr * ((self._epoch + 1) / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        elif self._epoch == self.warmup_epochs:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr
        else:
            self.cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"_epoch": self._epoch, "cosine": self.cosine.state_dict()}

    def load_state_dict(self, state):
        self._epoch = state["_epoch"]
        self.cosine.load_state_dict(state["cosine"])


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def get_teacher_forcing_ratio(epoch: int, config: dict) -> float:
    """Linearly decay teacher forcing from its initial value toward a minimum
    over `teacher_forcing_decay_epochs` (<= 0 keeps it constant)."""
    t = config["training"]
    start = t["teacher_forcing_ratio"]
    end = t.get("teacher_forcing_min", 0.0)
    decay_epochs = t.get("teacher_forcing_decay_epochs", 0)
    if decay_epochs and decay_epochs > 0:
        frac = min(1.0, max(0, epoch - 1) / decay_epochs)   # epoch is 1-indexed
        return start + (end - start) * frac
    return start


def train_one_epoch(model, loader, optimizer, config, epoch, device) -> dict:
    model.train()
    tf_ratio = get_teacher_forcing_ratio(epoch, config)
    grad_clip = config["training"]["grad_clip"]
    max_batches = config["training"].get("max_batches_per_epoch")

    agg, n_batches = {}, 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        if not bool(batch.get("target_mask", batch["mask"]).any()):
            continue
        optimizer.zero_grad()
        outputs = model(batch, teacher_forcing_ratio=tf_ratio)
        total_loss, loss_dict = compute_total_loss(outputs, batch, config, epoch)
        if not torch.isfinite(total_loss):
            print(f"  [warning] non-finite loss at epoch {epoch}; batch skipped")
            continue
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        for k, v in loss_dict.items():
            agg[k] = agg.get(k, 0.0) + v
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in agg.items()}


@torch.no_grad()
def validate(model, loader, config, epoch, device, seed: int = 12345) -> tuple:
    """Validation losses + minADE/minFDE at K=1 and K=num_predictions."""
    model.eval()
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        K = config["model"]["num_predictions"]
        agg, n_batches = {}, 0
        errors = {}
        for batch in loader:
            outputs = model(batch, teacher_forcing_ratio=0.0)
            _, loss_dict = compute_total_loss(outputs, batch, config, epoch)
            for k, v in loss_dict.items():
                agg[k] = agg.get(k, 0.0) + v
            n_batches += 1
            e = per_agent_errors(outputs["predictions"].cpu(), batch["future"],
                                 batch["target_mask"], batch.get("valid_mask"), ks=(1, K))
            for k, v in e.items():
                errors.setdefault(k, []).append(v)
    errors = {k: np.concatenate(v) for k, v in errors.items()}
    metrics = summarize_errors(errors, config["evaluation"].get("miss_threshold", 2.0))
    losses = {k: v / max(n_batches, 1) for k, v in agg.items()}
    return losses, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a trajectory prediction model")
    parser.add_argument("--config", type=str, default="configs/eth_ucy.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. model.use_map=false")
    parser.add_argument("--seed", type=int, default=None, help="Overrides training.seed")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Overrides training.checkpoint_path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume training from")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads (useful when running jobs in parallel)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.set)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.checkpoint is not None:
        config["training"]["checkpoint_path"] = args.checkpoint
    config.setdefault("evaluation", {})

    if args.threads:
        torch.set_num_threads(args.threads)
    seed = config["training"].get("seed", 0)
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] seed={seed} device={device}")

    if config["model"].get("type", "trajectory") == "cv":
        raise SystemExit("[train] The constant-velocity baseline has no parameters; "
                         "evaluate it directly with eval.py --set model.type=cv")

    # Datasets
    print(f"[train] dataset: {config['dataset']['type']}")
    train_dataset = DatasetFactory(config, split="train")
    val_dataset = DatasetFactory(config, split="val")
    num_workers = config["training"].get("num_workers", 0)
    pin_memory = device.type == "cuda"
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch_size = config["training"]["batch_size"]
    budget = config["training"].get("max_agents_per_batch")
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    generator=gen, max_agents_per_batch=budget)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  max_agents_per_batch=budget)
    print(f"[train] train: {len(train_dataset)} scenes | val: {len(val_dataset)} scenes")

    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model: {config['model'].get('type', 'trajectory')} ({n_params:,} params)")

    lr = config["training"]["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=config["training"].get("weight_decay", 0.0))
    total_epochs = config["training"]["epochs"]
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=config["training"].get("warmup_epochs", 5),
                                      total_epochs=total_epochs, base_lr=lr)

    wandb = try_import_wandb()
    use_wandb = config.get("logging", {}).get("use_wandb", False) and wandb is not None
    if use_wandb:
        wandb.init(project=config["logging"]["project_name"], config=config)

    ckpt_path = config["training"]["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    history_path = os.path.splitext(ckpt_path)[0] + "_history.json"

    K = config["model"]["num_predictions"]
    sel_keys = (f"min_ade_{K}", f"min_fde_{K}")
    patience = config["training"].get("early_stopping_patience", 15)
    best_score, epochs_no_improve, start_epoch = float("inf"), 0, 1
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score = ckpt.get("best_score", float("inf"))
        print(f"[train] resumed from {args.resume} at epoch {start_epoch}")

    log_every = config.get("logging", {}).get("log_every", 1)
    t_start = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, config, epoch, device)
        val_losses, metrics = validate(model, val_loader, config, epoch, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        score = metrics[sel_keys[0]] + metrics[sel_keys[1]]
        history.append({"epoch": epoch, "lr": lr_now, "time": time.time() - t0,
                        "train": train_losses, "val": val_losses, "metrics": metrics})

        if epoch % log_every == 0:
            m_str = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            print(f"[epoch {epoch:03d}/{total_epochs}] lr={lr_now:.1e} "
                  f"train={train_losses.get('total', 0):.3f} {m_str} ({time.time() - t0:.0f}s)",
                  flush=True)

        if score < best_score:
            best_score, epochs_no_improve = score, 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "val_metrics": metrics,
                "config": config,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] early stopping: no improvement for {patience} epochs")
                break

        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=1)
        if use_wandb:
            wandb.log({"epoch": epoch, "lr": lr_now,
                       **{f"train/{k}": v for k, v in train_losses.items()},
                       **{f"val/{k}": v for k, v in val_losses.items()},
                       **{f"metrics/{k}": v for k, v in metrics.items()}})

    if use_wandb:
        wandb.finish()
    print(f"[train] done in {(time.time() - t_start) / 60:.1f} min; best val "
          f"{sel_keys[0]}+{sel_keys[1]} = {best_score:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
