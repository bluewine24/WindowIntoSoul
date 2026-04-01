from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import DialogueExample


def _zero(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    effective_mask = mask if weights is None else (mask * weights)
    if effective_mask.sum() == 0:
        return _zero(prediction.device)
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    loss = loss * effective_mask
    return loss.sum() / effective_mask.sum()


def _label_aware_smoothness(latent: torch.Tensor, discrete_targets: torch.Tensor, discrete_mask: torch.Tensor) -> torch.Tensor:
    if latent.size(0) < 2:
        return _zero(latent.device)
    delta = latent[1:] - latent[:-1]
    magnitude = delta.pow(2).mean(dim=-1)

    pair_weights = torch.full((latent.size(0) - 1,), 0.6, device=latent.device)
    valid_pairs = discrete_mask[1:] & discrete_mask[:-1]
    same_label = discrete_targets[1:] == discrete_targets[:-1]
    pair_weights = torch.where(valid_pairs & same_label, torch.full_like(pair_weights, 1.0), pair_weights)
    pair_weights = torch.where(valid_pairs & (~same_label), torch.full_like(pair_weights, 0.35), pair_weights)
    return torch.mean(magnitude * pair_weights)


class EmotionTrajectoryLoss(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        training_cfg = config["training"]
        self.weights = training_cfg["loss_weights"]
        self.appraisal_stage_weights = training_cfg.get("appraisal_stage_weights", {}) or {}
        self.contrastive_margin = float(training_cfg["contrastive_margin"])
        self.label_smoothing = float(training_cfg["label_smoothing"])
        self.joint_gate_scale = float(training_cfg["joint_gate_scale"])

    def _appraisal_stage_weight(self, stage: str) -> float:
        return float(self.appraisal_stage_weights.get(stage, 1.0))

    def _build_targets(self, example: DialogueExample, role_to_id: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
        vad_targets = []
        vad_mask = []
        appraisal_targets = []
        appraisal_mask = []
        appraisal_confidence = []
        discrete_targets = []
        discrete_mask = []
        role_ids = []

        for turn in example.turns:
            role_ids.append(role_to_id.get(turn.role.lower(), role_to_id["other"]))

            if turn.labels.vad is None:
                vad_targets.append([0.0, 0.0, 0.0])
                vad_mask.append([0.0, 0.0, 0.0])
            else:
                current_vad_targets = []
                current_vad_mask = []
                for value in turn.labels.vad:
                    if value is None:
                        current_vad_targets.append(0.0)
                        current_vad_mask.append(0.0)
                    else:
                        current_vad_targets.append(float(value))
                        current_vad_mask.append(1.0)
                vad_targets.append(current_vad_targets)
                vad_mask.append(current_vad_mask)

            if turn.labels.appraisal is None:
                appraisal_targets.append([0.0] * 5)
                appraisal_mask.append([0.0] * 5)
                appraisal_confidence.append([0.0] * 5)
            else:
                current_appraisal_targets = []
                current_appraisal_mask = []
                current_appraisal_confidence = []
                raw_confidence = turn.labels.appraisal_confidence or [None] * len(turn.labels.appraisal)
                if len(raw_confidence) != len(turn.labels.appraisal):
                    raise ValueError(
                        f"Appraisal confidence must contain {len(turn.labels.appraisal)} values, got {len(raw_confidence)} "
                        f"for dialogue {example.dialogue_id}"
                    )
                for value in turn.labels.appraisal:
                    current_index = len(current_appraisal_targets)
                    confidence = raw_confidence[current_index]
                    if value is None:
                        current_appraisal_targets.append(0.0)
                        current_appraisal_mask.append(0.0)
                        current_appraisal_confidence.append(0.0)
                    else:
                        current_appraisal_targets.append(float(value))
                        current_appraisal_mask.append(1.0)
                        current_appraisal_confidence.append(max(0.0, min(1.0, float(confidence if confidence is not None else 1.0))))
                if len(current_appraisal_targets) != 5:
                    raise ValueError(
                        f"Appraisal labels must contain 5 values or null placeholders, got {len(current_appraisal_targets)} "
                        f"for dialogue {example.dialogue_id}"
                    )
                appraisal_targets.append(current_appraisal_targets)
                appraisal_mask.append(current_appraisal_mask)
                appraisal_confidence.append(current_appraisal_confidence)

            if turn.labels.discrete is None:
                discrete_targets.append(0)
                discrete_mask.append(False)
            else:
                discrete_targets.append(int(turn.labels.discrete))
                discrete_mask.append(True)

        return {
            "vad_targets": torch.tensor(vad_targets, dtype=torch.float32, device=device),
            "vad_mask": torch.tensor(vad_mask, dtype=torch.float32, device=device),
            "appraisal_targets": torch.tensor(appraisal_targets, dtype=torch.float32, device=device),
            "appraisal_mask": torch.tensor(appraisal_mask, dtype=torch.float32, device=device),
            "appraisal_confidence": torch.tensor(appraisal_confidence, dtype=torch.float32, device=device),
            "discrete_targets": torch.tensor(discrete_targets, dtype=torch.long, device=device),
            "discrete_mask": torch.tensor(discrete_mask, dtype=torch.bool, device=device),
            "role_ids": torch.tensor(role_ids, dtype=torch.long, device=device),
        }

    def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return _zero(logits.device)
        return F.cross_entropy(logits[mask], targets[mask], label_smoothing=self.label_smoothing)

    def _contrastive_loss(
        self,
        latents: torch.Tensor,
        discrete_targets: torch.Tensor,
        role_ids: torch.Tensor,
        discrete_mask: torch.Tensor,
    ) -> torch.Tensor:
        if discrete_mask.sum() < 3:
            return _zero(latents.device)

        z = F.normalize(latents[discrete_mask], dim=-1)
        labels = discrete_targets[discrete_mask]
        roles = role_ids[discrete_mask]
        distances = torch.cdist(z, z, p=2)
        losses = []

        for index in range(z.size(0)):
            positive_mask = (labels == labels[index]) & (roles == roles[index])
            positive_mask[index] = False
            negative_mask = labels != labels[index]
            if positive_mask.any() and negative_mask.any():
                positive_distance = distances[index][positive_mask].mean()
                negative_distance = distances[index][negative_mask].mean()
                losses.append(F.relu(positive_distance - negative_distance + self.contrastive_margin))

        if not losses:
            return _zero(latents.device)
        return torch.stack(losses).mean()

    def _consistency_loss(
        self,
        latents: torch.Tensor,
        discrete_targets: torch.Tensor,
        role_ids: torch.Tensor,
        discrete_mask: torch.Tensor,
    ) -> torch.Tensor:
        if discrete_mask.sum() < 2:
            return _zero(latents.device)

        z = latents[discrete_mask]
        labels = discrete_targets[discrete_mask]
        roles = role_ids[discrete_mask]
        distances = torch.cdist(z, z, p=2)
        losses = []

        for index in range(z.size(0)):
            positive_mask = (labels == labels[index]) & (roles == roles[index])
            positive_mask[index] = False
            if positive_mask.any():
                losses.append(distances[index][positive_mask].mean())

        if not losses:
            return _zero(latents.device)
        return torch.stack(losses).mean()

    def forward(self, model: nn.Module, outputs: list[Any], batch: list[DialogueExample], stage: str) -> dict[str, torch.Tensor]:
        device = outputs[0].z_emotion.device
        appraisal_stage_weight = self._appraisal_stage_weight(stage)
        metrics = {
            "vad": _zero(device),
            "appraisal": _zero(device),
            "discrete": _zero(device),
            "smoothness": _zero(device),
            "contrastive": _zero(device),
            "consistency": _zero(device),
            "stable_vad": _zero(device),
            "stable_appraisal": _zero(device),
            "stable_discrete": _zero(device),
            "gate_smoothness": _zero(device),
            "gate_fidelity": _zero(device),
        }

        pairwise_latents = []
        pairwise_labels = []
        pairwise_roles = []
        pairwise_masks = []

        for output, example in zip(outputs, batch):
            targets = self._build_targets(example, model.role_to_id, device=device)

            if stage in {"base", "joint"}:
                metrics["vad"] += _masked_regression_loss(output.vad, targets["vad_targets"], targets["vad_mask"])
                metrics["appraisal"] += _masked_regression_loss(
                    output.appraisal,
                    targets["appraisal_targets"],
                    targets["appraisal_mask"],
                    weights=targets["appraisal_confidence"],
                )
                metrics["discrete"] += self._cross_entropy(
                    output.discrete_logits, targets["discrete_targets"], targets["discrete_mask"]
                )
                metrics["smoothness"] += _label_aware_smoothness(
                    output.z_emotion, targets["discrete_targets"], targets["discrete_mask"]
                )
                pairwise_latents.append(output.z_emotion)
                pairwise_labels.append(targets["discrete_targets"])
                pairwise_roles.append(targets["role_ids"])
                pairwise_masks.append(targets["discrete_mask"])

            if stage in {"gate", "joint"}:
                stable_vad, stable_appraisal, stable_discrete = model.heads_from_latent(output.z_stable)
                metrics["stable_vad"] += _masked_regression_loss(
                    stable_vad, targets["vad_targets"], targets["vad_mask"]
                )
                metrics["stable_appraisal"] += _masked_regression_loss(
                    stable_appraisal,
                    targets["appraisal_targets"],
                    targets["appraisal_mask"],
                    weights=targets["appraisal_confidence"],
                )
                metrics["stable_discrete"] += self._cross_entropy(
                    stable_discrete, targets["discrete_targets"], targets["discrete_mask"]
                )
                metrics["gate_smoothness"] += _label_aware_smoothness(
                    output.z_stable, targets["discrete_targets"], targets["discrete_mask"]
                )
                if output.z_stable.size(0) > 1:
                    metrics["gate_fidelity"] += F.mse_loss(output.z_stable[1:], output.z_emotion.detach()[1:])
                if stage == "gate":
                    pairwise_latents.append(output.z_stable)
                    pairwise_labels.append(targets["discrete_targets"])
                    pairwise_roles.append(targets["role_ids"])
                    pairwise_masks.append(targets["discrete_mask"])

        divisor = max(len(batch), 1)
        for name in metrics:
            metrics[name] = metrics[name] / divisor

        if pairwise_latents:
            stacked_latents = torch.cat(pairwise_latents, dim=0)
            stacked_labels = torch.cat(pairwise_labels, dim=0)
            stacked_roles = torch.cat(pairwise_roles, dim=0)
            stacked_masks = torch.cat(pairwise_masks, dim=0)
            metrics["contrastive"] = self._contrastive_loss(
                stacked_latents, stacked_labels, stacked_roles, stacked_masks
            )
            metrics["consistency"] = self._consistency_loss(
                stacked_latents, stacked_labels, stacked_roles, stacked_masks
            )

        base_total = (
            (self.weights["vad"] * metrics["vad"])
            + (self.weights["appraisal"] * appraisal_stage_weight * metrics["appraisal"])
            + (self.weights["discrete"] * metrics["discrete"])
            + (self.weights["smoothness"] * metrics["smoothness"])
            + (self.weights["contrastive"] * metrics["contrastive"])
            + (self.weights["consistency"] * metrics["consistency"])
        )
        gate_total = (
            (self.weights["vad"] * metrics["stable_vad"])
            + (self.weights["appraisal"] * appraisal_stage_weight * metrics["stable_appraisal"])
            + (self.weights["discrete"] * metrics["stable_discrete"])
            + (self.weights["gate_smoothness"] * metrics["gate_smoothness"])
            + (self.weights["gate_fidelity"] * metrics["gate_fidelity"])
        )

        if stage == "base":
            total = base_total
        elif stage == "gate":
            total = gate_total
        else:
            total = base_total + (self.joint_gate_scale * gate_total)

        metrics["appraisal_stage_weight"] = torch.tensor(appraisal_stage_weight, dtype=torch.float32, device=device)
        metrics["total"] = total
        return metrics
