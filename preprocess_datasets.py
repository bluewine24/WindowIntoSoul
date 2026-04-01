from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml

from datasets import APPRAISAL_NAMES, DISCRETE_EMOTIONS, DialogueExample, DialogueTurn, TurnLabels, dialogue_from_dict, dialogue_to_dict


TARGET_EMOTION_TO_INDEX = {name: index for index, name in enumerate(DISCRETE_EMOTIONS)}

COMMON_DISCRETE_LABEL_MAP = {
    "neutral": "neutral",
    "no emotion": "neutral",
    "other": "neutral",
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "optimism": "joy",
    "approval": "joy",
    "sadness": "sadness",
    "sad": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "anger": "anger",
    "angry": "anger",
    "rage": "anger",
    "fear": "fear",
    "afraid": "fear",
    "nervousness": "fear",
    "nervous": "fear",
    "anxiety": "fear",
    "surprise": "surprise",
    "surprised": "surprise",
    "realization": "surprise",
    "disgust": "disgust",
    "disapproval": "disgust",
    "love": "love",
    "caring": "love",
    "affection": "love",
    "embarrassment": "embarrassment",
    "embarrassed": "embarrassment",
    "pride": "pride",
    "admiration": "pride",
    "guilt": "guilt",
    "remorse": "guilt",
    "relief": "relief",
    "gratitude": "relief",
    "curiosity": "curiosity",
    "curious": "curiosity",
    "confusion": "curiosity",
    "interest": "curiosity",
    "frustration": "frustration",
    "annoyance": "frustration",
    "annoyed": "frustration",
    "desire": "curiosity",
    "afraid": "fear",
    "anxious": "fear",
    "terrified": "fear",
    "apprehensive": "fear",
    "hopeful": "joy",
    "grateful": "relief",
    "joyful": "joy",
    "excited": "joy",
    "prepared": "pride",
    "confident": "pride",
    "content": "joy",
    "faithful": "love",
    "trusting": "love",
    "sentimental": "love",
    "nostalgic": "sadness",
    "lonely": "sadness",
    "furious": "anger",
    "disappointed": "sadness",
    "disgusted": "disgust",
    "embarrassed": "embarrassment",
    "impressed": "surprise",
    "jealous": "frustration",
    "devastated": "sadness",
    "caring": "love",
}

BUILTIN_LABEL_MAPS: dict[str, dict[str, str]] = {
    "default": COMMON_DISCRETE_LABEL_MAP,
    "goemotions": {
        **COMMON_DISCRETE_LABEL_MAP,
        "gratitude": "relief",
        "approval": "joy",
        "caring": "love",
        "admiration": "pride",
        "disappointment": "sadness",
        "desire": "curiosity",
        "confusion": "curiosity",
        "optimism": "joy",
    },
    "meld": COMMON_DISCRETE_LABEL_MAP,
    "empatheticdialogues": COMMON_DISCRETE_LABEL_MAP,
    "dailydialog": {
        **COMMON_DISCRETE_LABEL_MAP,
        "0": "neutral",
        "1": "anger",
        "2": "disgust",
        "3": "fear",
        "4": "joy",
        "5": "sadness",
        "6": "surprise",
    },
    "emobank": COMMON_DISCRETE_LABEL_MAP,
}


@dataclass
class RawTurn:
    speaker_id: str
    text: str
    order_index: int
    labels: TurnLabels


@dataclass
class RawDialogue:
    dataset_name: str
    dialogue_id: str
    turns: list[RawTurn]
    source_split: Optional[str] = None


class CharacterVectorFactory:
    def __init__(self, dim: int, strategy: str = "dataset_character", seed: int = 42):
        self.dim = dim
        self.strategy = strategy
        self.seed = seed
        self._cache: dict[str, list[float]] = {}

    def character_id(self, dataset_name: str, raw_dialogue_id: str, raw_character_id: str) -> str:
        if self.strategy == "sample_source":
            return f"{dataset_name}::{raw_dialogue_id}::{raw_character_id}"
        return f"{dataset_name}::{raw_character_id}"

    def vector_for(self, dataset_name: str, raw_dialogue_id: str, raw_character_id: str) -> list[float]:
        key = self.character_id(dataset_name, raw_dialogue_id, raw_character_id)
        if key not in self._cache:
            digest = hashlib.md5(f"{self.seed}:{key}".encode("utf-8")).hexdigest()
            rng = random.Random(int(digest, 16))
            values = [(rng.random() * 2.0) - 1.0 for _ in range(self.dim)]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            self._cache[key] = [round(value / norm, 6) for value in values]
        return self._cache[key]


def _normalize_split_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    split_aliases = {
        "train": "train",
        "training": "train",
        "trn": "train",
        "dev": "val",
        "valid": "val",
        "validation": "val",
        "val": "val",
        "test": "test",
        "testing": "test",
        "tst": "test",
    }
    return split_aliases.get(normalized)


def load_recipe(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_records(path: Path, file_format: str) -> Iterator[dict[str, Any]]:
    if file_format == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if file_format == "jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                raw = line.strip()
                if raw:
                    yield json.loads(raw)
        return
    if file_format == "json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            yield from payload
            return
        if isinstance(payload, dict):
            records = payload.get("records", payload.get("data"))
            if isinstance(records, list):
                yield from records
                return
        raise ValueError(f"Unsupported JSON payload in {path}")
    raise ValueError(f"Unsupported format: {file_format}")


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _coerce_confidence(value: Any) -> Optional[float]:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return max(0.0, min(1.0, float(numeric)))


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _is_positive_label(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "1.0", "true", "yes"}


def _parse_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        separators = ["|", ";", ","]
        for separator in separators:
            if separator in raw:
                return [item.strip() for item in raw.split(separator) if item.strip()]
        return [raw]
    return [value]


def _scale_to_range(value: Optional[float], source_range: Optional[list[float]], target_range: tuple[float, float] = (-1.0, 1.0)) -> Optional[float]:
    if value is None or source_range is None:
        return value
    source_min, source_max = float(source_range[0]), float(source_range[1])
    target_min, target_max = target_range
    if math.isclose(source_min, source_max):
        return value
    position = (value - source_min) / (source_max - source_min)
    scaled = target_min + position * (target_max - target_min)
    return round(float(scaled), 6)


def _resolve_discrete_map(spec: dict[str, Any]) -> dict[str, str]:
    mapping_name = str(spec.get("discrete_map", "default"))
    mapping = dict(BUILTIN_LABEL_MAPS.get(mapping_name, BUILTIN_LABEL_MAPS["default"]))
    custom_map = spec.get("custom_discrete_map", {}) or {}
    for key, value in custom_map.items():
        mapping[str(key).lower()] = str(value).lower()
    return mapping


def _map_discrete_label(value: Any, spec: dict[str, Any]) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        if 0 <= value < len(DISCRETE_EMOTIONS):
            return value
        return None

    mapping = _resolve_discrete_map(spec)
    labels = _parse_sequence(value)
    multi_label_policy = str(spec.get("multi_label_policy", "first_supported"))

    if len(labels) > 1 and multi_label_policy == "skip_if_multi":
        return None

    for raw_label in labels:
        normalized = str(raw_label).strip().lower()
        if normalized in TARGET_EMOTION_TO_INDEX:
            return TARGET_EMOTION_TO_INDEX[normalized]
        mapped = mapping.get(normalized)
        if mapped in TARGET_EMOTION_TO_INDEX:
            return TARGET_EMOTION_TO_INDEX[mapped]
        if normalized.isdigit():
            numeric = int(normalized)
            if 0 <= numeric < len(DISCRETE_EMOTIONS):
                return numeric
    return None


def _extract_vad(container: dict[str, Any], spec: dict[str, Any]) -> Optional[list[Optional[float]]]:
    values: list[Optional[float]]
    if spec.get("vad_field"):
        values = [_coerce_float(value) for value in _parse_sequence(container.get(spec["vad_field"]))]
    else:
        fields = spec.get("vad_fields", [])
        values = [_coerce_float(container.get(field)) for field in fields]
        if (not values or all(value is None for value in values)) and isinstance(container.get("vad"), list):
            values = [_coerce_float(value) for value in container.get("vad", [])]

    if not values:
        return None
    if len(values) != 3:
        raise ValueError(f"VAD requires exactly 3 values, got {values}")

    source_range = spec.get("vad_range")
    scaled = [_scale_to_range(value, source_range) for value in values]
    if all(value is None for value in scaled):
        return None
    return scaled


def _extract_appraisal(container: dict[str, Any], spec: dict[str, Any]) -> Optional[list[Optional[float]]]:
    values: dict[str, Optional[float]] = {}

    if spec.get("appraisal_field"):
        appraisal_payload = container.get(spec["appraisal_field"])
        if isinstance(appraisal_payload, dict):
            values = {name: _coerce_float(appraisal_payload.get(name)) for name in APPRAISAL_NAMES}
        else:
            sequence = [_coerce_float(value) for value in _parse_sequence(appraisal_payload)]
            if len(sequence) not in {0, len(APPRAISAL_NAMES)}:
                raise ValueError(f"Appraisal requires {len(APPRAISAL_NAMES)} values, got {sequence}")
            values = {name: sequence[index] for index, name in enumerate(APPRAISAL_NAMES)} if sequence else {}
    else:
        field_map = spec.get("appraisal_fields", {}) or {}
        nested_appraisal = container.get("appraisal") if isinstance(container.get("appraisal"), dict) else {}
        nested_labels = container.get("labels") if isinstance(container.get("labels"), dict) else {}
        for target_name in APPRAISAL_NAMES:
            raw_field = field_map.get(target_name)
            if raw_field and raw_field in container:
                values[target_name] = _coerce_float(container.get(raw_field))
            elif raw_field and raw_field in nested_appraisal:
                values[target_name] = _coerce_float(nested_appraisal.get(raw_field))
            elif raw_field and raw_field in nested_labels:
                values[target_name] = _coerce_float(nested_labels.get(raw_field))
            elif target_name in nested_appraisal:
                values[target_name] = _coerce_float(nested_appraisal.get(target_name))
            else:
                values[target_name] = None

    if not values:
        return None
    source_range = spec.get("appraisal_range")
    aligned = [_scale_to_range(values.get(name), source_range) for name in APPRAISAL_NAMES]
    if all(value is None for value in aligned):
        return None
    return aligned


def _extract_appraisal_confidence(
    container: dict[str, Any],
    spec: dict[str, Any],
    appraisal: Optional[list[Optional[float]]],
) -> Optional[list[Optional[float]]]:
    if appraisal is None:
        return None

    values: dict[str, Optional[float]] = {}

    if spec.get("appraisal_confidence_field"):
        confidence_payload = container.get(spec["appraisal_confidence_field"])
        if isinstance(confidence_payload, dict):
            values = {name: _coerce_confidence(confidence_payload.get(name)) for name in APPRAISAL_NAMES}
        else:
            sequence = [_coerce_confidence(value) for value in _parse_sequence(confidence_payload)]
            if len(sequence) not in {0, len(APPRAISAL_NAMES)}:
                raise ValueError(f"Appraisal confidence requires {len(APPRAISAL_NAMES)} values, got {sequence}")
            values = {name: sequence[index] for index, name in enumerate(APPRAISAL_NAMES)} if sequence else {}
    else:
        field_map = spec.get("appraisal_confidence_fields", {}) or {}
        nested_labels = container.get("labels") if isinstance(container.get("labels"), dict) else {}
        nested_appraisal = container.get("appraisal") if isinstance(container.get("appraisal"), dict) else {}
        nested_confidence = (
            container.get("appraisal_confidence") if isinstance(container.get("appraisal_confidence"), dict) else {}
        )
        for target_name in APPRAISAL_NAMES:
            raw_field = field_map.get(target_name)
            if raw_field and raw_field in container:
                values[target_name] = _coerce_confidence(container.get(raw_field))
            elif raw_field and raw_field in nested_confidence:
                values[target_name] = _coerce_confidence(nested_confidence.get(raw_field))
            elif raw_field and raw_field in nested_labels:
                values[target_name] = _coerce_confidence(nested_labels.get(raw_field))
            elif raw_field and raw_field in nested_appraisal:
                values[target_name] = _coerce_confidence(nested_appraisal.get(raw_field))
            elif target_name in nested_confidence:
                values[target_name] = _coerce_confidence(nested_confidence.get(target_name))
            elif target_name in nested_labels:
                values[target_name] = _coerce_confidence(nested_labels.get(target_name))
            else:
                values[target_name] = None

    if not values:
        return [0.0 if value is None else 1.0 for value in appraisal]

    aligned = []
    for appraisal_value, target_name in zip(appraisal, APPRAISAL_NAMES):
        confidence = values.get(target_name)
        if appraisal_value is None:
            aligned.append(0.0)
        elif confidence is None:
            aligned.append(1.0)
        else:
            aligned.append(confidence)
    return aligned


def _normalize_labels(container: dict[str, Any], spec: dict[str, Any]) -> TurnLabels:
    labels_source = container.get(spec.get("labels_field"), container) if spec.get("labels_field") else container
    if not isinstance(labels_source, dict):
        labels_source = container

    if spec.get("discrete_onehot_fields"):
        discrete_source: Any = [
            field
            for field in spec.get("discrete_onehot_fields", [])
            if _is_positive_label(labels_source.get(field, container.get(field)))
        ]
    else:
        discrete_source = labels_source.get(spec.get("discrete_field", "discrete"))

    appraisal = _extract_appraisal(labels_source, spec)
    return TurnLabels(
        vad=_extract_vad(labels_source, spec),
        appraisal=appraisal,
        appraisal_confidence=_extract_appraisal_confidence(labels_source, spec, appraisal),
        discrete=_map_discrete_label(discrete_source, spec),
    )


def _load_single_turn_dialogues(spec: dict[str, Any]) -> list[RawDialogue]:
    rows = list(_read_records(Path(spec["path"]), spec["format"]))
    text_field = spec["text_field"]
    dialogue_field = spec.get("dialogue_field")
    character_field = spec.get("character_field")
    speaker_field = spec.get("speaker_field")
    split_field = spec.get("split_field")
    dialogues = []
    for index, row in enumerate(rows):
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        dialogue_id = str(row.get(dialogue_field, f"{spec['name']}-{index}"))
        speaker_id = str(row.get(character_field) or row.get(speaker_field) or "self")
        dialogues.append(
            RawDialogue(
                dataset_name=spec["name"],
                dialogue_id=dialogue_id,
                turns=[RawTurn(speaker_id=speaker_id, text=text, order_index=0, labels=_normalize_labels(row, spec))],
                source_split=_normalize_split_name(row.get(split_field)) if split_field else None,
            )
        )
    return dialogues


def _load_dialogue_table(spec: dict[str, Any]) -> list[RawDialogue]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_records(Path(spec["path"]), spec["format"]):
        dialogue_id = str(row[spec["dialogue_field"]])
        grouped[dialogue_id].append(row)

    dialogues = []
    turn_field = spec.get("turn_field")
    speaker_field = spec.get("speaker_field", "speaker")
    text_field = spec["text_field"]

    for dialogue_id, rows in grouped.items():
        if turn_field:
            rows.sort(key=lambda item: float(item.get(turn_field, 0)))
        turns = []
        for row_index, row in enumerate(rows):
            text = str(row.get(text_field, "")).strip()
            if not text:
                continue
            turns.append(
                RawTurn(
                    speaker_id=str(row.get(speaker_field, f"speaker_{row_index % 2}")),
                    text=text,
                    order_index=row_index if turn_field is None else int(float(row.get(turn_field, row_index))),
                    labels=_normalize_labels(row, spec),
                )
            )
        if turns:
            dialogues.append(
                RawDialogue(
                    dataset_name=spec["name"],
                    dialogue_id=dialogue_id,
                    turns=turns,
                    source_split=_normalize_split_name(spec.get("fixed_split")),
                )
            )
    return dialogues


def _load_dialogue_jsonl(spec: dict[str, Any]) -> list[RawDialogue]:
    dialogues = []
    turns_field = spec.get("turns_field", "turns")
    text_field = spec.get("text_field", "text")
    speaker_field = spec.get("speaker_field", "speaker")

    for record_index, record in enumerate(_read_records(Path(spec["path"]), spec["format"])):
        raw_turns = record.get(turns_field, [])
        if not isinstance(raw_turns, list):
            continue
        dialogue_id = str(record.get(spec.get("dialogue_field", "dialogue_id"), f"{spec['name']}-{record_index}"))
        turns = []
        for turn_index, raw_turn in enumerate(raw_turns):
            text = str(raw_turn.get(text_field, "")).strip()
            if not text:
                continue
            turns.append(
                RawTurn(
                    speaker_id=str(raw_turn.get(speaker_field, f"speaker_{turn_index % 2}")),
                    text=text,
                    order_index=turn_index,
                    labels=_normalize_labels(raw_turn, spec),
                )
            )
        if turns:
            dialogues.append(
                RawDialogue(
                    dataset_name=spec["name"],
                    dialogue_id=dialogue_id,
                    turns=turns,
                    source_split=_normalize_split_name(spec.get("fixed_split")),
                )
            )
    return dialogues


def _parse_string_dialogue_list(raw_value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw_value)
    except Exception as exc:
        pattern = re.compile(r"""(['"])(.*?)(?<!\\)\1""", re.DOTALL)
        recovered = []
        for match in pattern.finditer(raw_value):
            recovered.append(ast.literal_eval(match.group(0)))
        if recovered:
            return [str(item).strip() for item in recovered if str(item).strip()]
        raise ValueError(f"Could not parse packed dialogue list: {raw_value[:120]}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Packed dialogue is not a list: {raw_value[:120]}")
    items = [str(item).strip() for item in parsed if str(item).strip()]
    if len(items) == 1:
        pattern = re.compile(r"""(['"])(.*?)(?<!\\)\1""", re.DOTALL)
        recovered = []
        for match in pattern.finditer(raw_value):
            recovered.append(ast.literal_eval(match.group(0)))
        if len(recovered) > 1:
            items = [str(item).strip() for item in recovered if str(item).strip()]
    return items


def _parse_numeric_bracket_list(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().lstrip("[").rstrip("]").strip()
    if not cleaned:
        return []
    return cleaned.split()


def _load_packed_dialogue_table(spec: dict[str, Any]) -> list[RawDialogue]:
    dialogues = []
    text_field = spec.get("text_field", "dialog")
    emotion_field = spec.get("emotion_field", "emotion")

    for index, row in enumerate(_read_records(Path(spec["path"]), spec["format"])):
        utterances = _parse_string_dialogue_list(str(row.get(text_field, "[]")))
        emotion_values = _parse_numeric_bracket_list(str(row.get(emotion_field, "[]")))
        turns = []
        for turn_index, utterance in enumerate(utterances):
            emotion_value = emotion_values[turn_index] if turn_index < len(emotion_values) else None
            labels_container = {spec.get("discrete_field", "discrete"): emotion_value}
            turns.append(
                RawTurn(
                    speaker_id=f"speaker_{turn_index % 2}",
                    text=utterance,
                    order_index=turn_index,
                    labels=_normalize_labels(labels_container, spec),
                )
            )
        if turns:
            dialogues.append(
                RawDialogue(
                    dataset_name=spec["name"],
                    dialogue_id=str(row.get(spec.get("dialogue_field", "dialogue_id"), f"{spec['name']}-{index}")),
                    turns=turns,
                    source_split=_normalize_split_name(spec.get("fixed_split")),
                )
            )
    return dialogues


_SPEAKER_SEGMENT_PATTERN = re.compile(r"(Customer|Agent)\s*:(.*?)(?=(?:Customer|Agent)\s*:|$)", re.DOTALL)


def _parse_empathetic_turns(prompt: str, response: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for speaker, content in _SPEAKER_SEGMENT_PATTERN.findall(prompt):
        text = content.strip()
        if text:
            turns.append((speaker.lower(), text))

    reply = response.strip()
    if reply:
        if not turns or turns[-1][0] != "agent" or turns[-1][1] != reply:
            turns.append(("agent", reply))
    return turns


def _load_empathetic_dialogues_csv(spec: dict[str, Any]) -> list[RawDialogue]:
    dialogues = []
    dialogue_field = spec.get("dialogue_field", "")
    prompt_field = spec.get("prompt_field", "empathetic_dialogues")
    response_field = spec.get("response_field", "labels")
    emotion_field = spec.get("discrete_field", "emotion")

    for index, row in enumerate(_read_records(Path(spec["path"]), spec["format"])):
        dialogue_id = str(row.get(dialogue_field, index)) if dialogue_field else f"{spec['name']}-{index}"
        prompt = str(row.get(prompt_field, "") or "")
        response = str(row.get(response_field, "") or "")
        parsed_turns = _parse_empathetic_turns(prompt, response)
        if not parsed_turns:
            continue

        turns = []
        for turn_index, (speaker_id, text) in enumerate(parsed_turns):
            labels_container: dict[str, Any] = {}
            if turn_index == 0:
                labels_container[emotion_field] = row.get(emotion_field)
            turns.append(
                RawTurn(
                    speaker_id=speaker_id,
                    text=text,
                    order_index=turn_index,
                    labels=_normalize_labels(labels_container, spec),
                )
            )
        dialogues.append(RawDialogue(dataset_name=spec["name"], dialogue_id=dialogue_id, turns=turns))
    return dialogues


def _load_unified_jsonl(spec: dict[str, Any], character_dim: int) -> list[DialogueExample]:
    examples = []
    for record in _read_records(Path(spec["path"]), spec["format"]):
        examples.append(dialogue_from_dict(record, character_dim=character_dim))
    return examples


def _raw_dialogues_from_spec(spec: dict[str, Any]) -> list[RawDialogue]:
    adapter = spec["adapter"]
    if adapter == "single_turn_table":
        return _load_single_turn_dialogues(spec)
    if adapter == "dialogue_table":
        return _load_dialogue_table(spec)
    if adapter == "dialogue_jsonl":
        return _load_dialogue_jsonl(spec)
    if adapter == "packed_dialogue_table":
        return _load_packed_dialogue_table(spec)
    if adapter == "empathetic_dialogues_csv":
        return _load_empathetic_dialogues_csv(spec)
    raise ValueError(f"Unsupported adapter for raw dialogue loading: {adapter}")


def _turn_distance(raw_turns: list[RawTurn], index: int, max_value: int) -> int:
    if index == 0:
        return 0
    distance = max(raw_turns[index].order_index - raw_turns[index - 1].order_index, 1)
    return min(distance, max_value)


def _target_character_samples(
    raw_dialogue: RawDialogue,
    vector_factory: CharacterVectorFactory,
    max_turn_distance: int,
) -> list[DialogueExample]:
    participants = []
    seen = set()
    for turn in raw_dialogue.turns:
        if turn.speaker_id not in seen:
            seen.add(turn.speaker_id)
            participants.append(turn.speaker_id)

    samples = []
    dialogue_id = f"{raw_dialogue.dataset_name}::{raw_dialogue.dialogue_id}"
    for participant in participants:
        turns = []
        for index, raw_turn in enumerate(raw_dialogue.turns):
            turns.append(
                DialogueTurn(
                    role="self" if raw_turn.speaker_id == participant else "other",
                    text=raw_turn.text,
                    turn_distance=_turn_distance(raw_dialogue.turns, index, max_turn_distance),
                    labels=raw_turn.labels,
                )
            )
        samples.append(
            DialogueExample(
                dialogue_id=dialogue_id,
                character_id=vector_factory.character_id(raw_dialogue.dataset_name, raw_dialogue.dialogue_id, participant),
                character_vector=vector_factory.vector_for(raw_dialogue.dataset_name, raw_dialogue.dialogue_id, participant),
                turns=turns,
            )
        )
    return samples


def _split_examples(examples: list[DialogueExample], split_cfg: dict[str, Any]) -> dict[str, list[DialogueExample]]:
    grouped: dict[str, list[DialogueExample]] = defaultdict(list)
    for example in examples:
        grouped[example.dialogue_id].append(example)

    dialogue_ids = list(grouped.keys())
    random.Random(int(split_cfg.get("seed", 42))).shuffle(dialogue_ids)

    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))
    train_cutoff = int(len(dialogue_ids) * train_ratio)
    val_cutoff = train_cutoff + int(len(dialogue_ids) * val_ratio)

    buckets = {"train": [], "val": [], "test": []}
    for index, dialogue_id in enumerate(dialogue_ids):
        if index < train_cutoff:
            target = "train"
        elif index < val_cutoff:
            target = "val"
        else:
            target = "test"
        buckets[target].extend(grouped[dialogue_id])
    return buckets


def _write_jsonl(path: Path, examples: Iterable[DialogueExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(dialogue_to_dict(example), ensure_ascii=True) + "\n")


def _collect_stats(splits: dict[str, list[DialogueExample]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split_name, examples in splits.items():
        vad_turns = 0
        appraisal_turns = 0
        discrete_turns = 0
        total_turns = 0
        datasets = defaultdict(int)
        for example in examples:
            dataset_name = example.dialogue_id.split("::", 1)[0] if "::" in example.dialogue_id else "unknown"
            datasets[dataset_name] += 1
            total_turns += len(example.turns)
            for turn in example.turns:
                if turn.labels.vad is not None and any(value is not None for value in turn.labels.vad):
                    vad_turns += 1
                if turn.labels.appraisal is not None and any(value is not None for value in turn.labels.appraisal):
                    appraisal_turns += 1
                if turn.labels.discrete is not None:
                    discrete_turns += 1
        stats[split_name] = {
            "samples": len(examples),
            "turns": total_turns,
            "vad_supervised_turns": vad_turns,
            "appraisal_supervised_turns": appraisal_turns,
            "discrete_supervised_turns": discrete_turns,
            "datasets": dict(sorted(datasets.items())),
        }
    return stats


def build_unified_dataset(recipe: dict[str, Any]) -> dict[str, list[DialogueExample]]:
    vector_cfg = recipe.get("character_vectors", {}) or {}
    output_cfg = recipe.get("output", {}) or {}
    turn_distance_cfg = recipe.get("turn_distance", {}) or {}
    vector_factory = CharacterVectorFactory(
        dim=int(vector_cfg.get("dim", 32)),
        strategy=str(vector_cfg.get("strategy", "dataset_character")),
        seed=int(vector_cfg.get("seed", 42)),
    )

    assigned_splits: dict[str, list[DialogueExample]] = {"train": [], "val": [], "test": []}
    unsplit_examples: list[DialogueExample] = []
    for spec in recipe.get("datasets", []):
        if not bool(spec.get("enabled", True)):
            continue

        dataset_path = Path(spec["path"])
        if not dataset_path.exists():
            if bool(spec.get("skip_if_missing", False)):
                print(f"[preprocess] Skipping missing dataset: {dataset_path}")
                continue
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        adapter = spec["adapter"]
        if adapter == "unified_jsonl":
            target_split = _normalize_split_name(spec.get("fixed_split"))
            loaded_examples = _load_unified_jsonl(spec, character_dim=vector_factory.dim)
            if target_split in assigned_splits:
                assigned_splits[target_split].extend(loaded_examples)
            else:
                unsplit_examples.extend(loaded_examples)
            continue

        raw_dialogues = _raw_dialogues_from_spec(spec)
        for raw_dialogue in raw_dialogues:
            examples = _target_character_samples(
                raw_dialogue=raw_dialogue,
                vector_factory=vector_factory,
                max_turn_distance=int(turn_distance_cfg.get("max_value", 15)),
            )
            target_split = raw_dialogue.source_split or _normalize_split_name(spec.get("fixed_split"))
            if target_split in assigned_splits:
                assigned_splits[target_split].extend(examples)
            else:
                unsplit_examples.extend(examples)

    random_splits = _split_examples(unsplit_examples, recipe.get("splits", {}) or {})
    splits = {
        "train": assigned_splits["train"] + random_splits["train"],
        "val": assigned_splits["val"] + random_splits["val"],
        "test": assigned_splits["test"] + random_splits["test"],
    }
    output_dir = Path(output_cfg.get("dir", "processed_data"))
    _write_jsonl(output_dir / output_cfg.get("train_file", "train.jsonl"), splits["train"])
    _write_jsonl(output_dir / output_cfg.get("val_file", "val.jsonl"), splits["val"])
    _write_jsonl(output_dir / output_cfg.get("test_file", "test.jsonl"), splits["test"])

    stats = _collect_stats(splits)
    with (output_dir / output_cfg.get("stats_file", "stats.json")).open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified target-character emotion datasets.")
    parser.add_argument("--config", default="preprocess_recipe.yaml", help="YAML recipe describing dataset adapters and split settings.")
    args = parser.parse_args()

    recipe = load_recipe(args.config)
    splits = build_unified_dataset(recipe)
    total = sum(len(examples) for examples in splits.values())
    print(f"Built {total} target-character samples.")
    for split_name, examples in splits.items():
        print(f"  {split_name}: {len(examples)} samples")


if __name__ == "__main__":
    main()
