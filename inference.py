from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from datasets import DialogueExample, dialogue_from_dict, load_dialogue_json
from text2emotion import TextToEmotionTrajectoryModel
from visualizer import TrajectoryVisualizer


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_default_dialogue(character_dim: int) -> DialogueExample:
    payload = {
        "dialogue_id": "inline-demo",
        "character_id": "inline-character",
        "character_vector": [0.0] * character_dim,
        "turns": [
            {"speaker": "inline-character", "role": "self", "text": "I am trying to keep it together.", "turn_distance": 0},
            {"speaker": "friend", "role": "other", "text": "You do not have to pretend with me.", "turn_distance": 1},
            {"speaker": "inline-character", "role": "self", "text": "That actually makes me feel calmer.", "turn_distance": 1},
        ],
    }
    return dialogue_from_dict(payload, character_dim=character_dim)


def load_checkpoint(model: TextToEmotionTrajectoryModel, checkpoint_path: str) -> bool:
    path = Path(checkpoint_path)
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a dialogue.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to a trained checkpoint.")
    parser.add_argument("--dialogue-path", default=None, help="Path to a single dialogue JSON file.")
    parser.add_argument("--output-path", default=None, help="Optional JSON output path.")
    parser.add_argument("--plot-path", default=None, help="Optional PNG output path.")
    parser.add_argument("--sanity", action="store_true", help="Run with random weights if no checkpoint is available.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextToEmotionTrajectoryModel(config)
    checkpoint_loaded = load_checkpoint(model, args.checkpoint)
    if not checkpoint_loaded and not args.sanity:
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Run training first or pass `--sanity` to use random weights."
        )
    model = model.to(device)

    if args.dialogue_path:
        dialogue = load_dialogue_json(args.dialogue_path, character_dim=int(config["model"]["character_dim"]))
    else:
        default_path = Path("sample_data/example_dialogue.json")
        if default_path.exists():
            dialogue = load_dialogue_json(str(default_path), character_dim=int(config["model"]["character_dim"]))
        else:
            dialogue = build_default_dialogue(character_dim=int(config["model"]["character_dim"]))

    model.eval()
    with torch.no_grad():
        output = model([dialogue], use_stable_history=True)[0]

    visualizer = TrajectoryVisualizer()
    visualizer.print_summary(dialogue, output)

    payload = output.to_dict(dialogue)
    output_path = args.output_path or config.get("inference", {}).get("output_path")
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved JSON output to {destination.resolve()}")

    if args.plot_path:
        visualizer.plot_dialogue(dialogue, output, save_path=args.plot_path)
        print(f"Saved plot to {Path(args.plot_path).resolve()}")

    if checkpoint_loaded:
        print(f"Used checkpoint: {Path(args.checkpoint).resolve()}")
    else:
        print("Used random model weights.")


if __name__ == "__main__":
    main()
