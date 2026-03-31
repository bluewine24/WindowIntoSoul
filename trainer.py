from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import DialogueDataset, collate_dialogues
from losses import EmotionTrajectoryLoss
from text2emotion import TextToEmotionTrajectoryModel


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: TextToEmotionTrajectoryModel, config: dict[str, Any], stage: str) -> AdamW:
    training_cfg = config["training"]
    if stage == "gate":
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return AdamW(params, lr=float(training_cfg["gate_lr"]), weight_decay=float(training_cfg["weight_decay"]))
    if stage == "joint":
        groups = model.parameter_groups(
            base_lr=float(training_cfg["joint_lr"]),
            encoder_lr=float(training_cfg["encoder_lr"]),
        )
        return AdamW(groups, weight_decay=float(training_cfg["weight_decay"]))
    params = model.parameter_groups(base_lr=float(training_cfg["lr"]))
    return AdamW(params, weight_decay=float(training_cfg["weight_decay"]))


def summarise_metrics(metrics: dict[str, float]) -> str:
    keys = ["total", "vad", "appraisal", "discrete", "smoothness", "gate_smoothness", "gate_fidelity"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)


def run_epoch(
    model: TextToEmotionTrajectoryModel,
    criterion: EmotionTrajectoryLoss,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    stage: str,
    gradient_clip_norm: float,
    limit_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    steps = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        if limit_batches is not None and batch_index > limit_batches:
            break

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(batch, use_stable_history=not training)
            losses = criterion(model, outputs, batch, stage=stage)
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu().item())
        steps += 1

    if steps == 0:
        return {"total": 0.0}
    return {name: value / steps for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: TextToEmotionTrajectoryModel,
    config: dict[str, Any],
    stage: str,
    epoch: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": stage,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def train(config: dict[str, Any], smoke_test: bool = False, limit_train_batches: int | None = None, limit_val_batches: int | None = None) -> None:
    set_seed(int(config["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DialogueDataset(
        path=config["data"]["train_path"],
        character_dim=int(config["model"]["character_dim"]),
    )
    val_dataset = DialogueDataset(
        path=config["data"]["val_path"],
        character_dim=int(config["model"]["character_dim"]),
    )

    batch_size = 1 if smoke_test else int(config["training"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_dialogues,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_dialogues,
    )

    model = TextToEmotionTrajectoryModel(config).to(device)
    criterion = EmotionTrajectoryLoss(config)
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    metrics_log = []
    best_val_loss = float("inf")

    stage_schedule = [
        ("base", 1 if smoke_test else int(config["training"]["base_epochs"])),
        ("gate", 1 if smoke_test else int(config["training"]["gate_epochs"])),
        ("joint", 1 if smoke_test else int(config["training"]["joint_epochs"])),
    ]

    gradient_clip_norm = float(config["training"]["gradient_clip_norm"])

    for stage, epochs in stage_schedule:
        if epochs <= 0:
            continue
        model.set_stage(stage)
        optimizer = build_optimizer(model, config, stage)
        print(f"Stage `{stage}` with {epochs} epoch(s).")

        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model=model,
                criterion=criterion,
                dataloader=train_loader,
                optimizer=optimizer,
                stage=stage,
                gradient_clip_norm=gradient_clip_norm,
                limit_batches=1 if smoke_test else limit_train_batches,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    criterion=criterion,
                    dataloader=val_loader,
                    optimizer=None,
                    stage=stage,
                    gradient_clip_norm=gradient_clip_norm,
                    limit_batches=1 if smoke_test else limit_val_batches,
                )

            metrics_log.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            print(f"  Epoch {epoch}: train[{summarise_metrics(train_metrics)}] val[{summarise_metrics(val_metrics)}]")

            save_checkpoint(
                path=checkpoint_dir / "latest.pt",
                model=model,
                config=config,
                stage=stage,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                save_checkpoint(
                    path=checkpoint_dir / "best.pt",
                    model=model,
                    config=config,
                    stage=stage,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_log, handle, indent=2)

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Artifacts written to {checkpoint_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dialogue-aware emotion trajectory model.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one small batch for each stage.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Optional cap for training batches per epoch.")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Optional cap for validation batches per epoch.")
    args = parser.parse_args()

    config = load_config(args.config)
    train(
        config=config,
        smoke_test=args.smoke_test,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )


if __name__ == "__main__":
    main()
