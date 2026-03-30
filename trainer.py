"""
Training loop for text2emotion.

Phases:
  Phase 1: valence + arousal only  (real labels, encoder frozen)
  Phase 2: + playfulness/shyness/affection  (weak labels added)
  Phase 3: visual inspection + iteration
  Phase 4: unfreeze top encoder layers, fine-tune
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, random_split
import yaml

from models.text2emotion import Text2EmotionModel, EmotionTrajectory, INTERPRETABLE_DIMS
from training.losses import Text2EmotionLoss
from data.datasets import UnifiedEmotionDataset, EvalSetDataset, EmotionSample
from evaluation.visualizer import TrajectoryVisualizer


# ---------------------------------------------------------------------------
# Collate function — handles variable clause counts
# ---------------------------------------------------------------------------

def collate_samples(batch: List[EmotionSample]):
    """
    Returns plain lists — the model handles tokenization internally.
    Labels are returned separately for loss computation.
    """
    texts  = [s.text  for s in batch]
    modes  = [s.label.mode for s in batch]
    labels = [s.label for s in batch]
    return texts, modes, labels


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Text2EmotionTrainer:

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self.visualizer = TrajectoryVisualizer()
        self.current_phase = 1

        Path(self.cfg["logging"]["save_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _build_model(self):
        mc = self.cfg["model"]
        self.model = Text2EmotionModel(
            encoder_name=    mc["encoder"],
            mode_dim=        mc["mode_dim"],
            gru_hidden=      mc["gru_hidden"],
            gru_layers=      mc["gru_layers"],
            gru_dropout=     mc["gru_dropout"],
            interpretable_dims=mc["interpretable_dims"],
            latent_dims=     mc["latent_dims"],
            freeze_encoder=  mc["freeze_encoder"],
        ).to(self.device)

    def _build_loss(self):
        lc = self.cfg["training"]["loss_weights"]
        self.criterion = Text2EmotionLoss(
            w_interpretable= lc["interpretable"],
            w_contrastive=   lc["contrastive"],
            w_smoothness=    lc["smoothness"],
            weak_label_weight=self.cfg["model"].get("weak_label_weight",
                              self.cfg["data"].get("weak_label_weight", 0.3)),
            temperature=     self.cfg["training"]["contrastive_temperature"],
        )

    def _build_optimizer(self):
        tc = self.cfg["training"]
        self.optimizer = AdamW(
            self.model.param_groups(tc["learning_rate"], tc["encoder_lr"]),
            weight_decay=tc["weight_decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5, verbose=True
        )

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_data(self):
        dc = self.cfg["data"]

        # Phase 1: train.csv — weak dims masked (valence/arousal only)
        train_path = dc.get("train_path", "emotion_data/train.csv")
        val_path   = dc.get("val_path",   "emotion_data/val.csv")

        if not Path(train_path).exists():
            raise FileNotFoundError(
                f"Training data not found: {train_path}\n"
                f"Run: python data/build_dataset.py"
            )

        # Phase 1: mask weak dims so only valence/arousal are supervised
        self.train_data_p1 = UnifiedEmotionDataset(train_path, label_mask_weak=True)
        # Phase 2: same data but with weak dims unmasked
        self.train_data_p2 = UnifiedEmotionDataset(train_path, label_mask_weak=False)
        self.val_data = UnifiedEmotionDataset(val_path, label_mask_weak=False)

        print(f"[Data] Train: {len(self.train_data_p1)} samples (phase 1, weak masked)")
        print(f"[Data] Train: {len(self.train_data_p2)} samples (phase 2, all dims)")
        print(f"[Data] Val:   {len(self.val_data)} samples")

        # Eval set — supports both old hand-curated and new build_dataset.py format
        eval_path = dc.get("eval_set_path", "emotion_data/eval.csv")
        if not Path(eval_path).exists():
            eval_path = dc.get("eval_set_path", "data/eval_set.csv")

        if Path(eval_path).exists():
            self.eval_set = EvalSetDataset(eval_path)
            print(f"[Data] Eval set: {len(self.eval_set)} samples")
        else:
            self.eval_set = None
            print("[Data] WARNING: No eval set found. Run `python data/datasets.py` to generate template.")

    def _make_loader(self, dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_samples,
            num_workers=2,
            pin_memory=self.device.type == "cuda",
        )

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def _train_step(self, texts, modes, labels) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Forward
        trajectories: List[EmotionTrajectory] = self.model(texts, modes)

        total_loss = torch.tensor(0.0, device=self.device)
        loss_log = {"total": 0., "interpretable": 0., "contrastive": 0., "smoothness": 0.}

        for traj, label in zip(trajectories, labels):
            label_tensor, mask_tensor = label.to_tensor()
            label_tensor = label_tensor.to(self.device)
            mask_tensor  = mask_tensor.to(self.device)

            # Broadcast label to all clauses (sentence-level label applied to all)
            M = traj.interpretable.shape[0]
            label_expanded = label_tensor.unsqueeze(0).expand(M, -1)
            mask_expanded  = mask_tensor.unsqueeze(0).expand(M, -1)

            losses = self.criterion(
                pred_interp=traj.interpretable,
                pred_latent=traj.latent,
                target_interp=label_expanded,
                label_mask=mask_expanded,
                positive_pairs=None,   # TODO: add pair mining
                phase=self.current_phase,
            )
            total_loss += losses["total"]
            for k in loss_log:
                loss_log[k] += losses[k].item()

        total_loss = total_loss / len(labels)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {k: v / len(labels) for k, v in loss_log.items()}

    # -----------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _val_step(self, texts, modes, labels) -> Dict[str, float]:
        self.model.eval()
        trajectories = self.model(texts, modes)

        loss_log = {"total": 0., "interpretable": 0., "contrastive": 0., "smoothness": 0.}
        for traj, label in zip(trajectories, labels):
            label_tensor, mask_tensor = label.to_tensor()
            label_tensor = label_tensor.to(self.device)
            mask_tensor  = mask_tensor.to(self.device)
            M = traj.interpretable.shape[0]
            losses = self.criterion(
                pred_interp=traj.interpretable,
                pred_latent=traj.latent,
                target_interp=label_tensor.unsqueeze(0).expand(M, -1),
                label_mask=mask_tensor.unsqueeze(0).expand(M, -1),
                phase=self.current_phase,
            )
            for k in loss_log:
                loss_log[k] += losses[k].item()

        return {k: v / len(labels) for k, v in loss_log.items()}

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self):
        self.load_data()
        tc = self.cfg["training"]

        best_val_loss = float("inf")
        patience_counter = 0
        log_path = Path(self.cfg["logging"]["log_dir"]) / "training_log.csv"

        with open(log_path, "w") as f:
            f.write("epoch,phase,train_total,train_interp,val_total,val_interp\n")

        for epoch in range(1, tc["max_epochs"] + 1):

            # --- Phase transitions ---
            if epoch == tc["phase1_epochs"] + 1:
                print("\n[Trainer] === Entering Phase 2: unmasking weak dims ===")
                self.current_phase = 2
                self.train_loader = self._make_loader(self.train_data_p2)
                print(f"[Trainer] Switched to full label dataset ({len(self.train_data_p2)} samples)")

            if epoch == 1:
                self.train_loader = self._make_loader(self.train_data_p1)
                self.val_loader   = self._make_loader(self.val_data, shuffle=False)

            # --- Train ---
            train_logs = []
            for texts, modes, labels in self.train_loader:
                log = self._train_step(texts, modes, labels)
                train_logs.append(log)

            train_avg = {k: sum(l[k] for l in train_logs) / len(train_logs)
                         for k in train_logs[0]}

            # --- Validate ---
            val_logs = []
            for texts, modes, labels in self.val_loader:
                log = self._val_step(texts, modes, labels)
                val_logs.append(log)

            val_avg = {k: sum(l[k] for l in val_logs) / len(val_logs)
                       for k in val_logs[0]}

            self.scheduler.step(val_avg["total"])

            print(f"Epoch {epoch:3d} | Phase {self.current_phase} | "
                  f"Train {train_avg['total']:.4f} (i={train_avg['interpretable']:.4f}) | "
                  f"Val {val_avg['total']:.4f} (i={val_avg['interpretable']:.4f})")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{self.current_phase},"
                        f"{train_avg['total']:.5f},{train_avg['interpretable']:.5f},"
                        f"{val_avg['total']:.5f},{val_avg['interpretable']:.5f}\n")

            # --- Visualize on eval set ---
            if self.eval_set and epoch % tc.get("eval_every_n_epochs", 2) == 0:
                self.run_eval_visualization(epoch)

            # --- Checkpoint ---
            if val_avg["total"] < best_val_loss:
                best_val_loss = val_avg["total"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_avg["total"], tag="best")
            else:
                patience_counter += 1
                if patience_counter >= tc["patience"]:
                    print(f"[Trainer] Early stopping at epoch {epoch}")
                    break

        print(f"[Trainer] Training complete. Best val loss: {best_val_loss:.5f}")

    # -----------------------------------------------------------------------
    # Eval visualization
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def run_eval_visualization(self, epoch: int = 0):
        """Run model on eval set and visualize trajectories."""
        if not self.eval_set:
            print("[Eval] No eval set loaded.")
            return

        self.model.eval()
        print(f"\n[Eval] Epoch {epoch} — trajectory preview:")

        for i, sample in enumerate(self.eval_set):
            traj = self.model([sample.text], [sample.label.mode])[0]
            self.visualizer.print_trajectory(
                text=sample.text,
                trajectory=traj,
                label=sample.label,
            )
            if i >= 14:   # show first 15
                break

        # Save plot
        save_path = Path(self.cfg["logging"]["log_dir"]) / f"trajectories_epoch{epoch:03d}.png"
        self.visualizer.plot_eval_set(self.model, self.eval_set, save_path=str(save_path))

    # -----------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = ""):
        path = Path(self.cfg["logging"]["save_dir"]) / f"model_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "phase": self.current_phase,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"[Trainer] Checkpoint saved: {path}  (val_loss={val_loss:.5f})")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.current_phase = ckpt.get("phase", 1)
        print(f"[Trainer] Loaded checkpoint from {path} (epoch={ckpt['epoch']}, phase={self.current_phase})")


if __name__ == "__main__":
    trainer = Text2EmotionTrainer()
    trainer.train()
