"""
Loss functions for text2emotion training.

Three components:
  1. Interpretable regression loss (MSE / SmoothL1)
  2. Latent contrastive loss
  3. Temporal smoothness loss  — dimension-aware weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-dimension smoothness weights
# ---------------------------------------------------------------------------

# Slow dims: shyness, affection should not spike
# Fast dims: arousal, surprise can spike freely
SMOOTHNESS_WEIGHTS = {
    "valence":     0.15,
    "arousal":     0.05,
    "playfulness": 0.10,
    "shyness":     0.25,
    "affection":   0.25,
}

# Phase-dependent loss weight for weak-supervised dims
WEAK_DIMS = {"playfulness", "shyness", "affection"}
WEAK_DIM_INDICES = [2, 3, 4]   # indices in the 5-dim interpretable vector


# ---------------------------------------------------------------------------
# 1. Interpretable regression loss
# ---------------------------------------------------------------------------

class InterpretableLoss(nn.Module):
    """
    MSE loss on interpretable dims.
    Applies lower weight to weak-supervised dims (playfulness/shyness/affection).
    """
    def __init__(self, weak_label_weight: float = 0.3):
        super().__init__()
        self.weak_weight = weak_label_weight

    def forward(self,
                pred: torch.Tensor,         # [M, 5]
                target: torch.Tensor,       # [M, 5]
                label_mask: Optional[torch.Tensor] = None,   # [M, 5] binary
                phase: int = 1) -> torch.Tensor:
        """
        Args:
            pred:        predicted [M, 5]
            target:      ground truth [M, 5]
            label_mask:  1 where label exists, 0 where unknown
            phase:       training phase — weak dims down-weighted in phase 1
        """
        loss = F.mse_loss(pred, target, reduction="none")   # [M, 5]

        # Down-weight weak dims in early phases
        if phase <= 2:
            weights = torch.ones(5, device=pred.device)
            for idx in WEAK_DIM_INDICES:
                weights[idx] = self.weak_weight
            loss = loss * weights.unsqueeze(0)

        # Apply label mask if provided (ignore unlabeled dims)
        if label_mask is not None:
            loss = loss * label_mask

        return loss.mean()


# ---------------------------------------------------------------------------
# 2. Latent contrastive loss (NT-Xent style)
# ---------------------------------------------------------------------------

class LatentContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss on latent dims.

    Positive pairs: same conversation / same mode (defined by pair_indices).
    Hard negatives: opposite valence/arousal direction.

    Temperature controls sharpness — lower = harder.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self,
                latent: torch.Tensor,           # [B, 8]  — batch of clause latents
                positive_pairs: torch.Tensor,   # [P, 2]  — index pairs that are similar
                ) -> torch.Tensor:
        """
        Args:
            latent:         [B, 8] latent vectors for a batch of clauses
            positive_pairs: [P, 2] index pairs of positive (similar) clauses
        """
        if positive_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=latent.device)

        # L2 normalize
        z = F.normalize(latent, dim=-1)         # [B, 8]

        # Cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temp      # [B, B]

        # Mask out diagonal (self-similarity)
        mask_diag = torch.eye(latent.shape[0], dtype=torch.bool, device=latent.device)
        sim = sim.masked_fill(mask_diag, -1e9)

        # For each positive pair, compute NT-Xent loss
        losses = []
        for i, j in positive_pairs:
            # i should predict j
            log_prob = sim[i] - torch.logsumexp(sim[i], dim=0)
            losses.append(-log_prob[j])

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 3. Temporal smoothness loss
# ---------------------------------------------------------------------------

class SmoothnessLoss(nn.Module):
    """
    Penalizes sharp changes between neighboring clause emotion states.
    Per-dimension weights: slow dims (shyness, affection) penalized more.
    Arousal/surprise can spike — penalized least.
    """
    def __init__(self, dim_weights: Optional[List[float]] = None):
        super().__init__()
        if dim_weights is None:
            # [valence, arousal, playfulness, shyness, affection]
            dim_weights = [
                SMOOTHNESS_WEIGHTS["valence"],
                SMOOTHNESS_WEIGHTS["arousal"],
                SMOOTHNESS_WEIGHTS["playfulness"],
                SMOOTHNESS_WEIGHTS["shyness"],
                SMOOTHNESS_WEIGHTS["affection"],
            ]
        self.register_buffer(
            "weights",
            torch.tensor(dim_weights, dtype=torch.float)
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: [M, D] — sequence of emotion vectors (interpretable dims only)
        Returns:
            scalar smoothness loss
        """
        if trajectory.shape[0] < 2:
            return torch.tensor(0.0, device=trajectory.device)

        delta = trajectory[1:] - trajectory[:-1]           # [M-1, D]
        delta_sq = delta.pow(2)                             # [M-1, D]

        n_dims = min(delta_sq.shape[-1], self.weights.shape[0])
        weighted = delta_sq[:, :n_dims] * self.weights[:n_dims].to(trajectory.device)

        return weighted.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class Text2EmotionLoss(nn.Module):
    """
    Combined loss for text2emotion training.

      L = w_interp * interp_loss
        + w_contrast * contrastive_loss
        + w_smooth * smoothness_loss
    """
    def __init__(self,
                 w_interpretable: float = 1.0,
                 w_contrastive: float = 0.3,
                 w_smoothness: float = 0.1,
                 weak_label_weight: float = 0.3,
                 temperature: float = 0.07):
        super().__init__()

        self.w_i = w_interpretable
        self.w_c = w_contrastive
        self.w_s = w_smoothness

        self.interp_loss    = InterpretableLoss(weak_label_weight)
        self.contrastive    = LatentContrastiveLoss(temperature)
        self.smoothness     = SmoothnessLoss()

    def forward(self,
                pred_interp: torch.Tensor,          # [M, 5]
                pred_latent: torch.Tensor,          # [M, 8]
                target_interp: torch.Tensor,        # [M, 5]
                label_mask: Optional[torch.Tensor] = None,
                positive_pairs: Optional[torch.Tensor] = None,
                phase: int = 1) -> Dict[str, torch.Tensor]:

        # 1. Regression on interpretable dims
        l_interp = self.interp_loss(pred_interp, target_interp, label_mask, phase)

        # 2. Contrastive on latent dims
        if positive_pairs is not None and positive_pairs.shape[0] > 0:
            l_contrast = self.contrastive(pred_latent, positive_pairs)
        else:
            l_contrast = torch.tensor(0.0, device=pred_interp.device)

        # 3. Smoothness on interpretable trajectory
        l_smooth = self.smoothness(pred_interp)

        total = (self.w_i * l_interp +
                 self.w_c * l_contrast +
                 self.w_s * l_smooth)

        return {
            "total":       total,
            "interpretable": l_interp,
            "contrastive": l_contrast,
            "smoothness":  l_smooth,
        }
