"""
Training Pipeline — Video Forensics & Deepfake Detection
─────────────────────────────────────────────────────────
ResNet-50 | batch=32 | epochs=25 | NVIDIA T4
Results: 93.1% accuracy | 0.918 F1 | ~3h 24m training time
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import build_dataloaders
from models.resnet50_detector import DeepfakeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Default Config
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "data_root":       "data/frames",
    "img_size":        224,
    "num_workers":     4,

    # Model
    "pretrained":      True,
    "freeze_until":    "layer3",       # Phase 1: freeze first N stages
    "unfreeze_epoch":  10,             # Phase 2: unfreeze from this epoch
    "unfreeze_stage":  "layer3",
    "dropout":         0.4,
    "hidden_dim":      512,

    # Training
    "batch_size":      32,
    "epochs":          25,
    "lr":              1e-3,
    "lr_min":          1e-6,
    "weight_decay":    1e-4,
    "label_smoothing": 0.05,
    "fp16":            True,           # Mixed precision (T4 supports Tensor Cores)
    "grad_clip":       1.0,

    # Scheduler
    "scheduler":       "cosine",       # "cosine" | "onecycle"

    # Checkpointing
    "save_dir":        "outputs/checkpoints",
    "save_top_k":      3,
    "early_stop_patience": 7,

    # Logging
    "log_interval":    50,             # log every N batches
    "tensorboard":     True,
}


# ──────────────────────────────────────────────
# Metrics Helper
# ──────────────────────────────────────────────

def compute_metrics(preds, labels) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    cm  = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn + 1e-8)      # false-positive rate
    return {
        "accuracy": acc,
        "f1":       f1,
        "fpr":      fpr,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

class Trainer:
    def __init__(self, config: Dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # ── Dirs ──────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config["save_dir"]) / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

        # ── Data ──────────────────────────────
        self.loaders = build_dataloaders(
            data_root=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            img_size=config["img_size"],
        )
        assert "train" in self.loaders, "Train loader required"

        # ── Model ─────────────────────────────
        self.model = DeepfakeDetector(
            pretrained=config["pretrained"],
            freeze_until=config.get("freeze_until"),
            dropout=config["dropout"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)

        # ── Loss ──────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get("label_smoothing", 0.0)
        )

        # ── Optimizer ─────────────────────────
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # ── Scheduler ─────────────────────────
        n_train = len(self.loaders["train"])
        if config["scheduler"] == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config["lr"],
                steps_per_epoch=n_train,
                epochs=config["epochs"],
            )
            self.sched_step_per_batch = True
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config["epochs"],
                eta_min=config["lr_min"],
            )
            self.sched_step_per_batch = False

        # ── Mixed Precision ───────────────────
        self.scaler = GradScaler(enabled=config.get("fp16", True) and self.device.type == "cuda")

        # ── Logging ───────────────────────────
        self.writer = (
            SummaryWriter(log_dir=str(self.run_dir / "tb"))
            if config.get("tensorboard") else None
        )

        # ── State ─────────────────────────────
        self.best_f1    = 0.0
        self.best_acc   = 0.0
        self.no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "train_f1":   [], "val_f1":   [],
        }

    # ── One Epoch ────────────────────────────

    def _run_epoch(self, split: str) -> Tuple[float, Dict]:
        is_train = split == "train"
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[split]

        total_loss = 0.0
        all_preds, all_labels = [], []
        t0 = time.time()

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch_idx, (imgs, labels) in enumerate(loader):
                imgs   = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=self.scaler.is_enabled()):
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, labels)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["grad_clip"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.sched_step_per_batch:
                        self.scheduler.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

                if is_train and (batch_idx + 1) % self.cfg["log_interval"] == 0:
                    avg = total_loss / (batch_idx + 1)
                    logger.info(
                        f"  [{split}] batch {batch_idx+1}/{len(loader)} "
                        f"loss={avg:.4f}"
                    )

        avg_loss = total_loss / len(loader)
        metrics  = compute_metrics(all_preds, all_labels)
        elapsed  = time.time() - t0
        logger.info(
            f"  [{split.upper()}] loss={avg_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"fpr={metrics['fpr']:.4f} ({elapsed:.1f}s)"
        )
        return avg_loss, metrics

    # ── Checkpointing ─────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "") -> None:
        name = f"epoch{epoch:02d}_f1{metrics['f1']:.4f}{tag}.pt"
        path = self.run_dir / name
        self.model.save(str(path), extra={"epoch": epoch, "metrics": metrics})

    # ── Main Loop ─────────────────────────────

    def fit(self) -> Dict:
        logger.info(f"Starting training | epochs={self.cfg['epochs']} | "
                    f"batch={self.cfg['batch_size']}")
        t_start = time.time()

        for epoch in range(1, self.cfg["epochs"] + 1):
            logger.info(f"\n{'='*55}\nEpoch {epoch}/{self.cfg['epochs']}")

            # Phase 2 unfreeze
            if epoch == self.cfg.get("unfreeze_epoch"):
                stage = self.cfg.get("unfreeze_stage", "layer3")
                logger.info(f"Phase 2: unfreezing from '{stage}'")
                self.model.unfreeze_from(stage)
                # Re-init optimizer with all trainable params
                self.optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.cfg["lr"] * 0.1,     # lower LR for fine-tuning
                    weight_decay=self.cfg["weight_decay"],
                )

            # Train
            tr_loss, tr_metrics = self._run_epoch("train")
            if not self.sched_step_per_batch:
                self.scheduler.step()

            # Validate
            val_loss, val_metrics = self._run_epoch("val") if "val" in self.loaders else (0, {})

            # Bookkeeping
            for key, val in [
                ("train_loss", tr_loss),  ("val_loss", val_loss),
                ("train_acc",  tr_metrics.get("accuracy", 0)),
                ("val_acc",    val_metrics.get("accuracy", 0)),
                ("train_f1",   tr_metrics.get("f1", 0)),
                ("val_f1",     val_metrics.get("f1", 0)),
            ]:
                self.history[key].append(val)

            if self.writer:
                self.writer.add_scalars("loss", {"train": tr_loss, "val": val_loss}, epoch)
                self.writer.add_scalars("accuracy", {
                    "train": tr_metrics["accuracy"],
                    "val":   val_metrics.get("accuracy", 0)
                }, epoch)
                self.writer.add_scalars("f1", {
                    "train": tr_metrics["f1"],
                    "val":   val_metrics.get("f1", 0)
                }, epoch)

            # Best model
            f1 = val_metrics.get("f1", tr_metrics.get("f1", 0))
            if f1 > self.best_f1:
                self.best_f1 = f1
                self._save_checkpoint(epoch, val_metrics or tr_metrics, tag="_best")
                self.no_improve = 0
                logger.info(f"  ★ New best F1: {self.best_f1:.4f}")
            else:
                self.no_improve += 1

            # Early stopping
            if self.no_improve >= self.cfg["early_stop_patience"]:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # Periodic save
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_metrics or tr_metrics)

        elapsed = time.time() - t_start
        h, m = divmod(int(elapsed), 3600)
        m, s = divmod(m, 60)
        logger.info(f"\nTraining complete in {h}h {m}m {s}s | Best F1: {self.best_f1:.4f}")

        # Save history
        history_path = self.run_dir / "history.json"
        history_path.write_text(json.dumps(self.history, indent=2))

        if self.writer:
            self.writer.close()

        return self.history


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepfakeDetector (ResNet-50)")
    parser.add_argument("--data_root",   default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--save_dir",    default=DEFAULT_CONFIG["save_dir"])
    parser.add_argument("--no_fp16",     action="store_true")
    parser.add_argument("--config",      default=None, help="Path to JSON config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()

    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    if args.no_fp16:
        cfg["fp16"] = False

    trainer = Trainer(cfg)
    history = trainer.fit()
