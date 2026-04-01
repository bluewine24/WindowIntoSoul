from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

from torch.utils.data import Dataset


DISCRETE_EMOTIONS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "love",
    "embarrassment",
    "pride",
    "guilt",
    "relief",
    "curiosity",
    "frustration",
]

APPRAISAL_NAMES = [
    "coping",
    "goal_relevance",
    "novelty",
    "pleasantness",
    "norm_fit",
]


@dataclass
class TurnLabels:
    vad: Optional[List[Optional[float]]] = None
    appraisal: Optional[List[Optional[float]]] = None
    appraisal_confidence: Optional[List[Optional[float]]] = None
    discrete: Optional[int] = None


@dataclass
class DialogueTurn:
    role: str
    text: str
    turn_distance: int
    labels: TurnLabels = field(default_factory=TurnLabels)


@dataclass
class DialogueExample:
    dialogue_id: str
    character_id: str
    character_vector: List[float]
    turns: List[DialogueTurn]


def _normalize_vector(vector: Iterable[float], dim: int) -> List[float]:
    values = [float(v) for v in vector]
    if len(values) >= dim:
        return values[:dim]
    return values + [0.0] * (dim - len(values))


def _normalize_optional_sequence(values: Optional[Iterable[Any]], dim: int, name: str) -> Optional[List[Optional[float]]]:
    if values is None:
        return None
    sequence = [None if value is None else float(value) for value in values]
    if len(sequence) != dim:
        raise ValueError(f"{name} must contain exactly {dim} values, got {len(sequence)}")
    return sequence


def _default_appraisal_confidence(appraisal: Optional[List[Optional[float]]]) -> Optional[List[Optional[float]]]:
    if appraisal is None:
        return None
    return [0.0 if value is None else 1.0 for value in appraisal]


def dialogue_from_dict(payload: dict[str, Any], character_dim: int = 64) -> DialogueExample:
    turns = []
    for index, raw_turn in enumerate(payload.get("turns", [])):
        labels = raw_turn.get("labels", {}) or {}
        vad = _normalize_optional_sequence(labels.get("vad"), dim=3, name="vad")
        appraisal = _normalize_optional_sequence(labels.get("appraisal"), dim=len(APPRAISAL_NAMES), name="appraisal")
        appraisal_confidence = _normalize_optional_sequence(
            labels.get("appraisal_confidence"),
            dim=len(APPRAISAL_NAMES),
            name="appraisal_confidence",
        )
        if appraisal is None:
            appraisal_confidence = None
        elif appraisal_confidence is None:
            appraisal_confidence = _default_appraisal_confidence(appraisal)
        else:
            appraisal_confidence = [
                0.0 if appraisal_value is None else max(0.0, min(1.0, float(confidence or 0.0)))
                for appraisal_value, confidence in zip(appraisal, appraisal_confidence)
            ]
        turns.append(
            DialogueTurn(
                role=str(raw_turn.get("role", "self")).lower(),
                text=str(raw_turn.get("text", "")).strip(),
                turn_distance=int(raw_turn.get("turn_distance", 0 if index == 0 else 1)),
                labels=TurnLabels(
                    vad=vad,
                    appraisal=appraisal,
                    appraisal_confidence=appraisal_confidence,
                    discrete=labels.get("discrete"),
                ),
            )
        )
    if not turns:
        raise ValueError("Each dialogue must contain at least one turn.")

    return DialogueExample(
        dialogue_id=str(payload.get("dialogue_id", "dialogue")),
        character_id=str(payload.get("character_id", "character")),
        character_vector=_normalize_vector(payload.get("character_vector", []), character_dim),
        turns=turns,
    )


def dialogue_to_dict(example: DialogueExample) -> dict[str, Any]:
    return {
        "dialogue_id": example.dialogue_id,
        "character_id": example.character_id,
        "character_vector": example.character_vector,
        "turns": [
            {
                "role": turn.role,
                "text": turn.text,
                "turn_distance": turn.turn_distance,
                "labels": {
                    "vad": turn.labels.vad,
                    "appraisal": turn.labels.appraisal,
                    "appraisal_confidence": turn.labels.appraisal_confidence,
                    "discrete": turn.labels.discrete,
                },
            }
            for turn in example.turns
        ],
    }


class DialogueDataset(Dataset[DialogueExample]):
    def __init__(self, path: str, character_dim: int = 64):
        self.path = Path(path)
        self.character_dim = character_dim
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}. "
                "Run `python datasets.py --write-sample-data` to generate starter files."
            )
        self.examples = self._load()

    def _load(self) -> List[DialogueExample]:
        examples: List[DialogueExample] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {self.path}") from exc
                examples.append(dialogue_from_dict(payload, character_dim=self.character_dim))
        if not examples:
            raise ValueError(f"No dialogue examples were found in {self.path}")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DialogueExample:
        return self.examples[index]


def collate_dialogues(batch: List[DialogueExample]) -> List[DialogueExample]:
    return batch


def load_dialogue_json(path: str, character_dim: int = 64) -> DialogueExample:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dialogue_from_dict(payload, character_dim=character_dim)


def save_dialogue_json(example: DialogueExample, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dialogue_to_dict(example), handle, indent=2)


def _vector_from_seed(seed: int, dim: int) -> List[float]:
    rng = random.Random(seed)
    values = []
    for _ in range(dim):
        values.append(round((rng.random() * 2.0) - 1.0, 4))
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 4) for value in values]


def _with_default_appraisal_confidence(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for turn in turns:
        labels = turn.get("labels", {}) or {}
        appraisal = labels.get("appraisal")
        if appraisal is not None and labels.get("appraisal_confidence") is None:
            labels["appraisal_confidence"] = _default_appraisal_confidence(appraisal)
        turn["labels"] = labels
    return turns


def _record(dialogue_id: str, character_id: str, seed: int, turns: list[dict[str, Any]], character_dim: int) -> dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "character_id": character_id,
        "character_vector": _vector_from_seed(seed, character_dim),
        "turns": _with_default_appraisal_confidence(turns),
    }


def _sample_records(character_dim: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = [
        _record(
            "train-001",
            "ava",
            11,
            [
                {
                    "role": "self",
                    "text": "I cannot believe we finally shipped it.",
                    "turn_distance": 0,
                    "labels": {"vad": [0.75, 0.55, 0.30], "appraisal": [0.80, 0.35, 0.70, 0.60, 0.45], "discrete": 9},
                },
                {
                    "role": "other",
                    "text": "You carried the last stretch. You should feel proud.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.55, 0.35, 0.20], "appraisal": [0.70, 0.15, 0.80, 0.55, 0.75], "discrete": 7},
                },
                {
                    "role": "self",
                    "text": "Honestly, I mostly feel relieved that the pressure is over.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.45, 0.20, 0.55], "appraisal": [0.65, 0.10, 0.75, 0.80, 0.50], "discrete": 11},
                },
            ],
            character_dim,
        ),
        _record(
            "train-002",
            "noah",
            17,
            [
                {
                    "role": "self",
                    "text": "I snapped at her in the meeting and now I feel awful.",
                    "turn_distance": 0,
                    "labels": {"vad": [-0.55, 0.45, -0.60], "appraisal": [-0.65, 0.30, 0.40, -0.35, -0.25], "discrete": 10},
                },
                {
                    "role": "other",
                    "text": "You can still apologize. That matters.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.10, -0.10, 0.20], "appraisal": [0.20, 0.10, 0.65, 0.55, 0.80], "discrete": 0},
                },
                {
                    "role": "self",
                    "text": "Yeah. I am scared, but I want to make it right.",
                    "turn_distance": 1,
                    "labels": {"vad": [-0.15, 0.35, -0.10], "appraisal": [0.05, 0.20, 0.35, 0.10, 0.65], "discrete": 4},
                },
            ],
            character_dim,
        ),
        _record(
            "train-003",
            "lena",
            23,
            [
                {
                    "role": "self",
                    "text": "Wait, what do you mean the contract vanished?",
                    "turn_distance": 0,
                    "labels": {"vad": [-0.25, 0.85, -0.20], "appraisal": [-0.55, 0.95, -0.30, -0.20, -0.10], "discrete": 5},
                },
                {
                    "role": "other",
                    "text": "I am still checking, but it looks like the wrong folder got synced.",
                    "turn_distance": 1,
                    "labels": {"vad": [-0.05, 0.30, -0.15], "appraisal": [-0.20, 0.55, 0.10, 0.15, 0.00], "discrete": 13},
                },
                {
                    "role": "self",
                    "text": "That is so frustrating. We needed it today.",
                    "turn_distance": 1,
                    "labels": {"vad": [-0.60, 0.70, -0.30], "appraisal": [-0.75, 0.40, 0.20, -0.45, -0.15], "discrete": 13},
                },
            ],
            character_dim,
        ),
        _record(
            "train-004",
            "sora",
            29,
            [
                {
                    "role": "self",
                    "text": "I thought I would be excited, but I just feel heavy.",
                    "turn_distance": 0,
                    "labels": {"vad": [-0.70, -0.35, -0.40], "appraisal": [-0.60, 0.20, 0.30, -0.30, -0.10], "discrete": 2},
                },
                {
                    "role": "other",
                    "text": "Do you want me to stay with you for a while?",
                    "turn_distance": 1,
                    "labels": {"vad": [0.20, -0.20, 0.40], "appraisal": [0.35, 0.10, 0.80, 0.55, 0.95], "discrete": 7},
                },
                {
                    "role": "self",
                    "text": "Yeah. That would help more than you know.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.05, -0.30, 0.60], "appraisal": [0.45, 0.05, 0.70, 0.65, 0.90], "discrete": 7},
                },
            ],
            character_dim,
        ),
        _record(
            "train-005",
            "mina",
            31,
            [
                {
                    "role": "self",
                    "text": "He remembered my favorite tea. That was unexpectedly sweet.",
                    "turn_distance": 0,
                    "labels": {"vad": [0.55, 0.20, 0.80], "appraisal": [0.70, 0.35, 0.60, 0.50, 0.95], "discrete": 7},
                },
                {
                    "role": "other",
                    "text": "You are blushing right now.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.20, 0.45, 0.65], "appraisal": [0.40, 0.50, 0.35, 0.25, 0.70], "discrete": 8},
                },
                {
                    "role": "self",
                    "text": "Please do not make me talk about it.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.10, 0.35, 0.35], "appraisal": [0.10, 0.15, 0.45, 0.20, 0.55], "discrete": 8},
                },
            ],
            character_dim,
        ),
        _record(
            "train-006",
            "rae",
            37,
            [
                {
                    "role": "self",
                    "text": "Huh. I did not expect the prototype to move that smoothly.",
                    "turn_distance": 0,
                    "labels": {"vad": [0.30, 0.55, 0.10], "appraisal": [0.55, 0.90, 0.35, 0.45, 0.20], "discrete": 12},
                },
                {
                    "role": "other",
                    "text": "Do you want to take it apart and see why it worked?",
                    "turn_distance": 1,
                    "labels": {"vad": [0.25, 0.40, 0.05], "appraisal": [0.45, 0.65, 0.40, 0.55, 0.10], "discrete": 12},
                },
                {
                    "role": "self",
                    "text": "Absolutely. Now I really want to know what changed.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.55, 0.60, 0.05], "appraisal": [0.65, 0.55, 0.35, 0.70, 0.15], "discrete": 12},
                },
            ],
            character_dim,
        ),
    ]

    val = [
        _record(
            "val-001",
            "hara",
            41,
            [
                {
                    "role": "self",
                    "text": "I was sure they would laugh at me.",
                    "turn_distance": 0,
                    "labels": {"vad": [-0.35, 0.50, 0.10], "appraisal": [-0.20, 0.35, 0.10, -0.25, 0.10], "discrete": 8},
                },
                {
                    "role": "other",
                    "text": "They did not. They were impressed.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.30, 0.20, 0.15], "appraisal": [0.55, 0.20, 0.75, 0.55, 0.50], "discrete": 9},
                },
                {
                    "role": "self",
                    "text": "I know. It still feels unreal.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.20, 0.20, 0.10], "appraisal": [0.45, 0.55, 0.30, 0.35, 0.30], "discrete": 5},
                },
            ],
            character_dim,
        ),
        _record(
            "val-002",
            "tess",
            43,
            [
                {
                    "role": "self",
                    "text": "I am trying to stay calm, but I am angry.",
                    "turn_distance": 0,
                    "labels": {"vad": [-0.45, 0.55, -0.15], "appraisal": [-0.65, 0.20, 0.70, -0.20, -0.15], "discrete": 3},
                },
                {
                    "role": "other",
                    "text": "Take a breath. Tell me exactly what happened.",
                    "turn_distance": 1,
                    "labels": {"vad": [0.05, -0.10, 0.10], "appraisal": [0.15, 0.15, 0.80, 0.65, 0.35], "discrete": 0},
                },
                {
                    "role": "self",
                    "text": "Okay. If I explain it slowly, maybe I will stop shaking.",
                    "turn_distance": 1,
                    "labels": {"vad": [-0.10, 0.20, -0.05], "appraisal": [0.10, 0.25, 0.60, 0.35, 0.25], "discrete": 4},
                },
            ],
            character_dim,
        ),
    ]

    example_dialogue = _record(
        "demo-001",
        "demo_character",
        101,
        [
            {
                "role": "self",
                "text": "I thought this presentation would go badly.",
                "turn_distance": 0,
                "labels": {"vad": [-0.20, 0.40, -0.05], "appraisal": [0.00, 0.25, 0.30, 0.20, 0.15], "discrete": 4},
            },
            {
                "role": "other",
                "text": "You looked steady from the outside.",
                "turn_distance": 1,
                "labels": {"vad": [0.15, -0.10, 0.10], "appraisal": [0.25, 0.15, 0.75, 0.50, 0.45], "discrete": 0},
            },
            {
                "role": "self",
                "text": "Hearing that actually makes me feel relieved.",
                "turn_distance": 1,
                "labels": {"vad": [0.45, 0.10, 0.35], "appraisal": [0.60, 0.10, 0.80, 0.70, 0.60], "discrete": 11},
            },
        ],
        character_dim,
    )
    return train, val, example_dialogue


def write_sample_data(output_dir: str = "sample_data", character_dim: int = 64) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_records, val_records, example_dialogue = _sample_records(character_dim)

    with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    with (root / "val.jsonl").open("w", encoding="utf-8") as handle:
        for record in val_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    with (root / "example_dialogue.json").open("w", encoding="utf-8") as handle:
        json.dump(example_dialogue, handle, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dialogue dataset utilities.")
    parser.add_argument("--write-sample-data", action="store_true", help="Create starter train/val/demo files.")
    parser.add_argument("--output-dir", default="sample_data", help="Destination directory for generated files.")
    parser.add_argument("--character-dim", type=int, default=64, help="Dimension of the generated character vectors.")
    args = parser.parse_args()

    if args.write_sample_data:
        write_sample_data(output_dir=args.output_dir, character_dim=args.character_dim)
        print(f"Wrote sample data to {Path(args.output_dir).resolve()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
