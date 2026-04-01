from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from datasets import APPRAISAL_NAMES, DISCRETE_EMOTIONS, DialogueExample, dialogue_from_dict, dialogue_to_dict


APPRAISAL_DESCRIPTIONS = {
    "coping": "How able the target character seems to manage, influence, or recover from the situation.",
    "goal_relevance": "How important or consequential the situation appears for the target character's goals.",
    "novelty": "How unexpected, sudden, or unfamiliar the situation feels.",
    "pleasantness": "How positive versus negative the situation feels to the target character.",
    "norm_fit": "How acceptable, fair, or norm-consistent the situation feels.",
}

EMOTION_PRIORS: dict[str, tuple[list[float], float]] = {
    "neutral": ([0.0, 0.0, 0.0, 0.0, 0.0], 0.35),
    "joy": ([0.45, 0.55, 0.15, 0.80, 0.35], 0.50),
    "sadness": ([-0.45, 0.35, 0.10, -0.75, -0.05], 0.50),
    "anger": ([-0.70, 0.60, 0.30, -0.70, -0.45], 0.55),
    "fear": ([-0.60, 0.65, 0.55, -0.65, -0.20], 0.55),
    "surprise": ([0.00, 0.40, 0.85, 0.05, 0.00], 0.45),
    "disgust": ([-0.65, 0.45, 0.25, -0.80, -0.55], 0.55),
    "love": ([0.55, 0.50, 0.10, 0.75, 0.50], 0.50),
    "embarrassment": ([-0.30, 0.55, 0.35, -0.20, -0.10], 0.50),
    "pride": ([0.70, 0.45, 0.20, 0.65, 0.40], 0.50),
    "guilt": ([-0.50, 0.50, 0.20, -0.55, -0.45], 0.55),
    "relief": ([0.60, 0.25, 0.35, 0.60, 0.20], 0.50),
    "curiosity": ([0.30, 0.55, 0.70, 0.25, 0.10], 0.45),
    "frustration": ([-0.55, 0.60, 0.20, -0.55, -0.25], 0.55),
}

POSITIVE_HINTS = ("glad", "happy", "love", "great", "sweet", "relieved", "proud", "excited", "good")
NEGATIVE_HINTS = ("sad", "angry", "upset", "awful", "bad", "terrible", "frustrating", "scared", "hurt")
NOVELTY_HINTS = ("sudden", "unexpected", "surprised", "wait", "what", "wow", "did not expect")
GOAL_HINTS = ("need", "important", "must", "today", "deadline", "matter", "goal", "ship")
COPING_POSITIVE_HINTS = ("can", "able", "handle", "fix", "recover", "apologize", "solve", "help")
COPING_NEGATIVE_HINTS = ("cannot", "can't", "stuck", "helpless", "lost", "failed")
NORM_NEGATIVE_HINTS = ("wrong", "shouldn't", "unfair", "rude", "guilty", "sorry", "snapped")
NORM_POSITIVE_HINTS = ("fair", "kind", "okay", "right", "thanks", "apologize")


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _default_appraisal_confidence(appraisal: list[float | None]) -> list[float]:
    return [0.0 if value is None else 1.0 for value in appraisal]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_annotation(payload: dict[str, Any]) -> tuple[list[float | None], list[float]]:
    if "appraisal" in payload and isinstance(payload["appraisal"], dict):
        payload = payload["appraisal"]

    values: list[float | None] = []
    confidences: list[float] = []

    for name in APPRAISAL_NAMES:
        item = payload.get(name)
        if item is None:
            values.append(None)
            confidences.append(0.0)
            continue

        if isinstance(item, dict):
            raw_value = item.get("value")
            raw_confidence = item.get("confidence")
        else:
            raw_value = item
            raw_confidence = 0.5

        if raw_value is None:
            values.append(None)
            confidences.append(0.0)
            continue

        values.append(round(_clip(float(raw_value), -1.0, 1.0), 6))
        confidences.append(round(_clip(float(raw_confidence if raw_confidence is not None else 0.5), 0.0, 1.0), 6))

    return values, confidences


def _turn_context(dialogue: DialogueExample, turn_index: int, context_turns: int) -> str:
    start = max(0, turn_index - context_turns)
    window = dialogue.turns[start : turn_index + 1]
    lines = []
    for relative_index, turn in enumerate(window, start=start):
        prefix = "CURRENT" if relative_index == turn_index else "PRIOR"
        lines.append(f"{prefix} turn {relative_index} | role={turn.role} | text={turn.text}")
    return "\n".join(lines)


def _annotation_prompt(dialogue: DialogueExample, turn_index: int, context_turns: int) -> tuple[str, str]:
    turn = dialogue.turns[turn_index]
    system_prompt = (
        "You annotate appraisal dimensions for a target character in dialogue. "
        "Return only valid JSON. For each appraisal dimension, output an object with "
        "`value` in [-1, 1] and `confidence` in [0, 1]. Use null only if the evidence is truly absent. "
        "Scores should reflect the target character's internal interpretation of the current turn, "
        "using recent dialogue context and the current turn role relative to the target character.\n\n"
        "Dimensions:\n"
        + "\n".join(f"- {name}: {description}" for name, description in APPRAISAL_DESCRIPTIONS.items())
    )
    user_prompt = (
        f"Target character id: {dialogue.character_id}\n"
        f"Dialogue id: {dialogue.dialogue_id}\n"
        f"Current turn index: {turn_index}\n"
        f"Current turn role relative to target: {turn.role}\n\n"
        "Recent context:\n"
        f"{_turn_context(dialogue, turn_index, context_turns)}\n\n"
        "Return JSON with this shape exactly:\n"
        "{\n"
        '  "coping": {"value": 0.0, "confidence": 0.0},\n'
        '  "goal_relevance": {"value": 0.0, "confidence": 0.0},\n'
        '  "novelty": {"value": 0.0, "confidence": 0.0},\n'
        '  "pleasantness": {"value": 0.0, "confidence": 0.0},\n'
        '  "norm_fit": {"value": 0.0, "confidence": 0.0}\n'
        "}\n"
    )
    return system_prompt, user_prompt


class BaseAnnotator:
    def annotate(self, dialogue: DialogueExample, turn_index: int, context_turns: int) -> tuple[list[float | None], list[float]]:
        raise NotImplementedError


class MockAnnotator(BaseAnnotator):
    def annotate(self, dialogue: DialogueExample, turn_index: int, context_turns: int) -> tuple[list[float | None], list[float]]:
        del context_turns
        turn = dialogue.turns[turn_index]
        label_name = None
        if turn.labels.discrete is not None and 0 <= turn.labels.discrete < len(DISCRETE_EMOTIONS):
            label_name = DISCRETE_EMOTIONS[turn.labels.discrete]

        base_values, base_confidence = EMOTION_PRIORS.get(label_name or "neutral", EMOTION_PRIORS["neutral"])
        values = list(base_values)
        confidence = base_confidence
        text = turn.text.lower()

        if any(token in text for token in POSITIVE_HINTS):
            values[3] += 0.15
            confidence += 0.05
        if any(token in text for token in NEGATIVE_HINTS):
            values[3] -= 0.15
            confidence += 0.05
        if any(token in text for token in NOVELTY_HINTS):
            values[2] += 0.20
            confidence += 0.05
        if any(token in text for token in GOAL_HINTS):
            values[1] += 0.15
            confidence += 0.05
        if any(token in text for token in COPING_POSITIVE_HINTS):
            values[0] += 0.15
            confidence += 0.05
        if any(token in text for token in COPING_NEGATIVE_HINTS):
            values[0] -= 0.20
            confidence += 0.05
        if any(token in text for token in NORM_NEGATIVE_HINTS):
            values[4] -= 0.20
            confidence += 0.05
        if any(token in text for token in NORM_POSITIVE_HINTS):
            values[4] += 0.10
            confidence += 0.05

        values = [round(_clip(value, -1.0, 1.0), 6) for value in values]
        confidences = [round(_clip(confidence, 0.0, 0.95), 6)] * len(APPRAISAL_NAMES)
        return values, confidences


class OpenAICompatibleAnnotator(BaseAnnotator):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def annotate(self, dialogue: DialogueExample, turn_index: int, context_turns: int) -> tuple[list[float | None], list[float]]:
        system_prompt, user_prompt = _annotation_prompt(dialogue, turn_index, context_turns)
        body = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Annotation request failed with HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Annotation request failed: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected annotation response payload: {json.dumps(payload)[:500]}") from exc
        return _coerce_annotation(_extract_json_object(content))


def _load_annotator(args: argparse.Namespace) -> BaseAnnotator:
    if args.provider == "mock":
        return MockAnnotator()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable `{args.api_key_env}` is required for provider `{args.provider}`.")

    return OpenAICompatibleAnnotator(
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        timeout_s=args.timeout_s,
    )


def _needs_annotation(dialogue: DialogueExample, turn_index: int, overwrite: bool) -> bool:
    if overwrite:
        return True
    turn = dialogue.turns[turn_index]
    if turn.labels.appraisal is None:
        return True
    return any(value is None for value in turn.labels.appraisal)


def _merge_annotation(dialogue: DialogueExample, turn_index: int, values: list[float | None], confidences: list[float], overwrite: bool) -> None:
    turn = dialogue.turns[turn_index]
    existing_values = list(turn.labels.appraisal or [None] * len(APPRAISAL_NAMES))
    existing_confidence = list(
        turn.labels.appraisal_confidence
        or _default_appraisal_confidence(existing_values)
    )

    merged_values: list[float | None] = []
    merged_confidences: list[float] = []
    for index, existing_value in enumerate(existing_values):
        if overwrite or existing_value is None:
            merged_values.append(values[index])
            merged_confidences.append(confidences[index] if values[index] is not None else 0.0)
        else:
            merged_values.append(existing_value)
            merged_confidences.append(existing_confidence[index] if existing_value is not None else 0.0)

    turn.labels.appraisal = merged_values
    turn.labels.appraisal_confidence = merged_confidences


def annotate_file(
    input_path: str,
    output_path: str,
    annotator: BaseAnnotator,
    context_turns: int,
    limit: int | None,
    overwrite: bool,
    sleep_s: float,
    report_path: str | None,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_path": str(source),
        "output_path": str(destination),
        "dialogues_written": 0,
        "turns_seen": 0,
        "turns_annotated": 0,
        "turns_skipped": 0,
    }

    with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for dialogue_index, raw_line in enumerate(reader):
            if limit is not None and dialogue_index >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue

            example = dialogue_from_dict(json.loads(line))
            for turn_index in range(len(example.turns)):
                stats["turns_seen"] += 1
                if not _needs_annotation(example, turn_index, overwrite):
                    stats["turns_skipped"] += 1
                    continue

                values, confidences = annotator.annotate(example, turn_index, context_turns)
                _merge_annotation(example, turn_index, values, confidences, overwrite=overwrite)
                stats["turns_annotated"] += 1

                if sleep_s > 0:
                    time.sleep(sleep_s)

            writer.write(json.dumps(dialogue_to_dict(example), ensure_ascii=True) + "\n")
            stats["dialogues_written"] += 1

    if report_path:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        with report_destination.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate dialogue JSONL with synthetic appraisal values and confidence.")
    parser.add_argument("--input-path", required=True, help="Unified dialogue JSONL to annotate.")
    parser.add_argument("--output-path", required=True, help="Destination JSONL with synthetic appraisal labels.")
    parser.add_argument("--provider", choices=["mock", "openai_compatible"], default="mock", help="Annotation backend.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name for the OpenAI-compatible backend.")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="Base URL for an OpenAI-compatible API.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--context-turns", type=int, default=3, help="How many prior turns to include before the current turn.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on dialogues to annotate.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing appraisal values instead of filling gaps only.")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Optional delay between remote requests.")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="HTTP timeout for remote annotation requests.")
    parser.add_argument("--report-path", default=None, help="Optional JSON report with annotation counts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first annotation prompt and exit without writing.")
    args = parser.parse_args()

    annotator = _load_annotator(args)

    if args.dry_run:
        with Path(args.input_path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                example = dialogue_from_dict(json.loads(line))
                system_prompt, user_prompt = _annotation_prompt(example, turn_index=0, context_turns=args.context_turns)
                print("=== SYSTEM PROMPT ===")
                print(system_prompt)
                print("=== USER PROMPT ===")
                print(user_prompt)
                return
        raise ValueError(f"No dialogue records found in {args.input_path}")

    stats = annotate_file(
        input_path=args.input_path,
        output_path=args.output_path,
        annotator=annotator,
        context_turns=args.context_turns,
        limit=args.limit,
        overwrite=args.overwrite,
        sleep_s=args.sleep_s,
        report_path=args.report_path,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
