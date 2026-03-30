"""
Trajectory visualizer.
The most important debugging tool — lets you see if model outputs "feel right."
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import torch

from models.text2emotion import EmotionTrajectory, INTERPRETABLE_DIMS


# ---------------------------------------------------------------------------
# Terminal visualization (no deps, always works)
# ---------------------------------------------------------------------------

BAR_WIDTH = 20
DIM_COLORS = {
    "valence":     "\033[32m",   # green
    "arousal":     "\033[31m",   # red
    "playfulness": "\033[35m",   # magenta
    "shyness":     "\033[34m",   # blue
    "affection":   "\033[33m",   # yellow
}
RESET = "\033[0m"


def _bar(value: float, width: int = BAR_WIDTH) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


class TrajectoryVisualizer:

    def print_trajectory(self,
                         text: str,
                         trajectory: EmotionTrajectory,
                         label=None):
        """Print a clause-by-clause trajectory to terminal."""
        print(f"\n{'─'*60}")
        print(f"  TEXT: \"{text}\"")
        if label:
            print(f"  MODE: {label.mode}")
        print(f"{'─'*60}")

        for ci, clause in enumerate(trajectory.clause_texts):
            vals = trajectory.interpretable[ci].detach().cpu().tolist()
            print(f"\n  Clause {ci+1}: \"{clause}\"")
            for di, dim in enumerate(INTERPRETABLE_DIMS):
                v = vals[di]
                color = DIM_COLORS.get(dim, "")
                bar = _bar(v)
                label_str = ""
                if label:
                    true_v = getattr(label, dim)
                    if true_v is not None:
                        diff = v - true_v
                        sign = "+" if diff >= 0 else ""
                        label_str = f"  [true={true_v:.2f}, err={sign}{diff:.2f}]"
                print(f"    {dim:<12} {color}{bar}{RESET} {v:.3f}{label_str}")

        print()

    def compare_sentences(self, model, sentence_pairs: List[tuple], device=None):
        """
        Quick sanity check: compare emotion trajectories of sentence pairs.
        Usage:
            visualizer.compare_sentences(model, [
                ("hehe no way", "SPEAKING"),
                ("oh. okay.",   "REACTING"),
            ])
        """
        model.eval()
        texts = [p[0] for p in sentence_pairs]
        modes = [p[1] if len(p) > 1 else "SPEAKING" for p in sentence_pairs]

        with torch.no_grad():
            trajectories = model(texts, modes)

        for (text, *_), traj in zip(sentence_pairs, trajectories):
            self.print_trajectory(text=text, trajectory=traj)

    def plot_eval_set(self, model, eval_set, save_path: str = "trajectories.png"):
        """
        Plot a grid of trajectory bar charts for the eval set.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            print("[Visualizer] matplotlib not installed. Skipping plot.")
            return

        model.eval()
        n = min(len(eval_set), 15)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]

        colors = ["#4CAF50", "#F44336", "#9C27B0", "#2196F3", "#FF9800"]

        with torch.no_grad():
            for i in range(n):
                sample = eval_set[i]
                traj = model([sample.text], [sample.label.mode])[0]
                ax = axes[i]

                vals = traj.interpretable.detach().cpu().numpy()   # [M, 5]
                M = vals.shape[0]
                x = np.arange(M)
                width = 0.15

                for di, (dim, color) in enumerate(zip(INTERPRETABLE_DIMS, colors)):
                    offset = (di - 2) * width
                    ax.bar(x + offset, vals[:, di], width, label=dim,
                           color=color, alpha=0.8)

                    # Plot ground truth as horizontal line if available
                    true_v = getattr(sample.label, dim)
                    if true_v is not None:
                        ax.axhline(y=true_v, color=color, linestyle='--',
                                   alpha=0.4, linewidth=1)

                ax.set_title(f'"{sample.text}"  [{sample.label.mode}]',
                             fontsize=9, loc='left')
                ax.set_ylim(0, 1)
                ax.set_xticks(x)
                ax.set_xticklabels([f"C{j+1}" for j in range(M)], fontsize=7)
                ax.set_ylabel("intensity", fontsize=7)
                ax.tick_params(axis='both', labelsize=7)
                if i == 0:
                    ax.legend(loc='upper right', fontsize=7, ncol=5)

        plt.suptitle("Emotion Trajectories (bars=pred, dashed=truth)",
                     fontsize=11, y=1.002)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved trajectory plot: {save_path}")

    def plot_training_log(self, log_path: str = "runs/training_log.csv"):
        """Plot train/val loss curves from log CSV."""
        try:
            import matplotlib.pyplot as plt
            import csv
        except ImportError:
            print("[Visualizer] matplotlib not installed.")
            return

        epochs, train_loss, val_loss, phases = [], [], [], []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_total"]))
                val_loss.append(float(row["val_total"]))
                phases.append(int(row["phase"]))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, train_loss, label="Train", color="#2196F3")
        ax.plot(epochs, val_loss,   label="Val",   color="#F44336")

        # Mark phase transitions
        for i in range(1, len(phases)):
            if phases[i] != phases[i-1]:
                ax.axvline(x=epochs[i], color="gray", linestyle="--", alpha=0.7)
                ax.text(epochs[i]+0.2, max(train_loss)*0.9,
                        f"Phase {phases[i]}", fontsize=8, color="gray")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Text2Emotion Training Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = Path(log_path).parent / "loss_curve.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"[Visualizer] Loss curve saved: {out}")
