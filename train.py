"""
train.py

Purpose:
    Main training script for the trajectory prediction model.
    All hyperparameters are read from config.yaml.
    Uses DatasetFactory so switching datasets requires only a config change.

Usage:
    python train.py --config config.yaml
"""

import argparse
import os
import time
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import DatasetFactory, build_dataloader
from models.trajectory import TrajectoryModel
from utils.losses import compute_total_loss
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Optional wandb — gracefully skipped if not installed
# ---------------------------------------------------------------------------
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
    """
    Linear warmup for `warmup_epochs`, then cosine annealing.
    Wraps CosineAnnealingLR.
    """

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

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            lr = self.base_lr * (self._epoch / max(self.warmup_epochs, 1))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_epoch(epoch: int, total_epochs: int, loss_dict: dict,
              metrics: dict, lr: float, elapsed: float):
    """Print a formatted log line for each epoch."""
    loss_str = "  ".join(
        f"{k}={v:.4f}" for k, v in loss_dict.items() if k != "kl_weight"
    )
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(
        f"[Epoch {epoch:03d}/{total_epochs}]  "
        f"lr={lr:.2e}  time={elapsed:.1f}s\n"
        f"  LOSSES: {loss_str}\n"
        f"  METRICS: {metric_str}"
    )


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TrajectoryModel,
    loader,
    optimizer: optim.Optimizer,
    config: dict,
    epoch: int,
    device: torch.device,
) -> dict:
    model.train()
    teacher_forcing_ratio = config["training"]["teacher_forcing_ratio"]
    grad_clip = config["training"]["grad_clip"]

    agg_losses = {}
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)

        # Loss computation
        total_loss, loss_dict = compute_total_loss(outputs, batch, config, epoch)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            agg_losses[k] = agg_losses.get(k, 0.0) + v
        n_batches += 1

    # Average over batches
    avg_losses = {k: v / max(n_batches, 1) for k, v in agg_losses.items()}
    return avg_losses


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: TrajectoryModel,
    loader,
    config: dict,
    epoch: int,
    device: torch.device,
) -> tuple:
    model.eval()

    agg_losses = {}
    all_preds, all_probs, all_gt, all_masks = [], [], [], []
    n_batches = 0

    for batch in loader:
        outputs = model(batch, teacher_forcing_ratio=0.0)
        _, loss_dict = compute_total_loss(outputs, batch, config, epoch)

        for k, v in loss_dict.items():
            agg_losses[k] = agg_losses.get(k, 0.0) + v
        n_batches += 1

        all_preds.append(outputs["predictions"].cpu())          # (B, N, K, T, 2)
        all_probs.append(outputs["log_probs"].cpu())            # (B, N, K)
        all_gt.append(batch["future"].cpu())                    # (B, N, T, 2)
        all_masks.append(batch["mask"].cpu())                   # (B, N)

    avg_losses = {k: v / max(n_batches, 1) for k, v in agg_losses.items()}

    # Compute metrics on full validation set
    preds = torch.cat(all_preds, dim=0)                        # (N_total, N, K, T, 2)
    probs = torch.cat(all_probs, dim=0)                        # (N_total, N, K)
    gt    = torch.cat(all_gt,    dim=0)                        # (N_total, N, T, 2)
    masks = torch.cat(all_masks, dim=0)                        # (N_total, N)

    metrics = compute_all_metrics(preds.numpy(), probs.numpy(), gt.numpy(), masks.numpy())
    return avg_losses, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train trajectory prediction model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # Datasets
    print(f"[train] Loading dataset: {config['dataset']['type']}")
    train_dataset = DatasetFactory(config, split="train")
    val_dataset   = DatasetFactory(config, split="val")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    print(f"[train] Train: {len(train_dataset)} scenes | Val: {len(val_dataset)} scenes")

    # Model
    model = TrajectoryModel(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    total_epochs  = config["training"]["epochs"]
    warmup_epochs = 5

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        base_lr=config["training"]["learning_rate"],
    )

    # Optional wandb
    wandb = try_import_wandb()
    use_wandb = config["logging"].get("use_wandb", False) and wandb is not None
    if use_wandb:
        wandb.init(project=config["logging"]["project_name"], config=config)
        print("[train] Weights & Biases logging enabled")

    # Checkpoint path
    ckpt_path = config["training"]["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else ".", exist_ok=True)

    best_fde = float("inf")
    log_every = config["logging"].get("log_every", 1)
    patience = config["training"].get("early_stopping_patience", 15)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n[train] Starting training for {total_epochs} epochs\n")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, optimizer, config, epoch, device)
        val_losses, metrics = validate(model, val_loader, config, epoch, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        if epoch % log_every == 0:
            log_epoch(epoch, total_epochs, val_losses, metrics, current_lr, elapsed)
            
            # Print overfit comparison
            train_total = train_losses.get('total', 0)
            val_total = val_losses.get('total', 0)
            print(f"  TRAIN LOSS: {train_total:.4f} | VAL LOSS: {val_total:.4f}")
            
            if train_total > 0:
                ratio = val_total / train_total
                if ratio > 1.3:
                    print(f"  [warning] Potential overfitting: Val loss is {ratio:.2f}x Train loss.")

        # Early stopping based on validation loss
        current_val_loss = val_losses.get("total", float("inf"))
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [early stopping] Validation loss hasn't improved for {patience} epochs. Stopping.")
                break

        # Save best checkpoint
        if metrics["fde"] < best_fde:
            best_fde = metrics["fde"]
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "best_fde":     best_fde,
                "config":       config,
            }, ckpt_path)
            print(f"  [checkpoint] Saved best model (FDE={best_fde:.4f}) → {ckpt_path}")

        # Optional wandb logging
        if use_wandb:
            log_data = {
                "epoch": epoch,
                "lr": current_lr,
                **{f"train/{k}": v for k, v in train_losses.items()},
                **{f"val/{k}": v for k, v in val_losses.items()},
                **{f"metrics/{k}": v for k, v in metrics.items()},
            }
            try:
                wandb.log(log_data)
            except Exception:
                pass

    print(f"\n[train] Training complete. Best FDE: {best_fde:.4f}")
    print(f"[train] Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()