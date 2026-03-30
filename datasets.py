"""
datasets.py — PyTorch Dataset classes for text2emotion training.

PIPELINE POSITION:
  build_dataset.py  →  [train.csv / val.csv / eval.csv]  →  datasets.py  →  trainer.py

build_dataset.py writes clause-level CSVs with schema:
  sample_id, row_in_sample, mode, full_text, clause_text,
  valence, arousal, playfulness, shyness, affection,
  source, label_quality, notes

This file reads those CSVs and groups them back into sample-level objects
that the trainer and model expect.

KEY DESIGN DECISION:
  The model does its own clause segmentation internally (segmenter.py).
  So we feed it `full_text`, NOT `clause_text`.
  The clause-level labels from build_dataset.py are averaged back to
  sentence level for supervision — the model learns to produce trajectories
  that match the average label, with smoothness loss handling the temporal shape.

  Why not feed clause_text directly?
  - Single clauses produce trajectories of length 1 — no temporal signal
  - The model's own segmenter handles boundary detection
  - Sentence-level full_text preserves context for RoBERTa
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from models.text2emotion import INTERPRETABLE_DIMS


# ---------------------------------------------------------------------------
# Label + Sample containers (unchanged — trainer depends on these)
# ---------------------------------------------------------------------------

@dataclass
class EmotionLabel:
    """
    Sentence-level continuous label.
    None = label not available for this dim (masked out in loss).
    """
    valence:     Optional[float] = None
    arousal:     Optional[float] = None
    playfulness: Optional[float] = None
    shyness:     Optional[float] = None
    affection:   Optional[float] = None
    mode:        str = "SPEAKING"
    source:      str = "unknown"
    label_quality: str = "unknown"

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (label [5], mask [5]). Mask=0 where label is None."""
        vals, mask = [], []
        for dim in INTERPRETABLE_DIMS:
            v = getattr(self, dim)
            if v is not None:
                vals.append(float(v))
                mask.append(1.0)
            else:
                vals.append(0.0)
                mask.append(0.0)
        return torch.tensor(vals, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32)

    def is_weak(self) -> bool:
        return self.label_quality in ("weak", "synthetic_llm",
                                      "real_core_weak_aux",
                                      "real_core_heuristic_aux")


@dataclass
class EmotionSample:
    text:  str           # full_text — fed to model for segmentation
    label: EmotionLabel


# ---------------------------------------------------------------------------
# Core unified loader — reads build_dataset.py output format
# ---------------------------------------------------------------------------

class UnifiedEmotionDataset(Dataset):
    """
    Reads train.csv / val.csv produced by build_dataset.py.

    Schema expected (from build_dataset.py):
      sample_id, row_in_sample, mode, full_text, clause_text,
      valence, arousal, playfulness, shyness, affection,
      source, label_quality, notes

    Grouping strategy:
      Rows with the same sample_id are grouped.
      full_text is taken from the first clause row (they're all identical).
      Emotion labels are averaged across clauses within the sample.
      This gives one EmotionSample per original sentence.

    label_mask_weak:
      If True, dims from weak sources (synthetic, heuristic) are masked out
      of the label tensor — the loss ignores them.
      Set False only in Phase 2+ when you want weak supervision.
    """

    def __init__(self, csv_path: str, label_mask_weak: bool = False):
        self.samples: List[EmotionSample] = []
        self._load(csv_path, label_mask_weak)

    def _load(self, path: str, label_mask_weak: bool):
        df = pd.read_csv(path)
        self._validate_schema(df, path)

        # Clip emotion dims to [0, 1]
        for dim in ["valence", "arousal", "playfulness", "shyness", "affection"]:
            if dim in df.columns:
                df[dim] = pd.to_numeric(df[dim], errors="coerce").clip(0.0, 1.0)

        # Group by sample_id — each group is one sentence
        for sample_id, group in df.groupby("sample_id", sort=False):
            group = group.sort_values("row_in_sample")
            first = group.iloc[0]

            full_text = str(first.get("full_text", "")).strip()
            if not full_text:
                continue

            mode          = str(first.get("mode", "SPEAKING")).strip()
            source        = str(first.get("source", "unknown")).strip()
            label_quality = str(first.get("label_quality", "unknown")).strip()
            is_weak_src   = label_quality in _WEAK_SOURCES

            def avg_dim(col: str) -> Optional[float]:
                if col not in group.columns:
                    return None
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                return float(vals.mean()) if len(vals) > 0 else None

            label = EmotionLabel(
                valence=     avg_dim("valence"),
                arousal=     avg_dim("arousal"),
                # Mask weak dims in Phase 1 if flag is set
                playfulness= None if (label_mask_weak and is_weak_src) else avg_dim("playfulness"),
                shyness=     None if (label_mask_weak and is_weak_src) else avg_dim("shyness"),
                affection=   None if (label_mask_weak and is_weak_src) else avg_dim("affection"),
                mode=mode,
                source=source,
                label_quality=label_quality,
            )

            self.samples.append(EmotionSample(text=full_text, label=label))

    @staticmethod
    def _validate_schema(df: pd.DataFrame, path: str):
        required = ["sample_id", "full_text", "mode", "source", "label_quality"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{Path(path).name} is missing columns: {missing}\n"
                f"Run build_dataset.py first to generate correctly formatted CSVs."
            )

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# Sources considered "weak" — cute dims masked in Phase 1
_WEAK_SOURCES = {
    "synthetic_llm", "synthetic", "weak",
    "real_core_weak_aux", "real_core_heuristic_aux",
}


# ---------------------------------------------------------------------------
# Eval set — reads build_dataset.py eval.csv OR the old hand-curated format
# ---------------------------------------------------------------------------

class EvalSetDataset(Dataset):
    """
    Reads eval.csv (from build_dataset.py) or the old hand-curated eval_set.csv.

    Supports both schemas automatically:
      New (build_dataset.py): sample_id, full_text, clause_text, mode, ...
      Old (hand-curated):     text, valence, arousal, ..., mode, notes
    """

    def __init__(self, path: str):
        self.samples: List[EmotionSample] = []
        self.notes:   List[str] = []
        df = pd.read_csv(path)

        if "full_text" in df.columns:
            self._load_new_format(df)
        elif "text" in df.columns:
            self._load_old_format(df)
        else:
            raise ValueError(f"Eval CSV must have 'full_text' or 'text' column: {path}")

    def _load_new_format(self, df: pd.DataFrame):
        """Grouped by sample_id, uses full_text."""
        for _, group in df.groupby("sample_id", sort=False):
            if "row_in_sample" in group.columns:
                group = group.sort_values("row_in_sample")
            first = group.iloc[0]
            text = str(first.get("full_text", "")).strip()
            if not text:
                continue

            def avg(col):
                if col not in group.columns: return None
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                return float(vals.mean()) if len(vals) else None

            self.samples.append(EmotionSample(
                text=text,
                label=EmotionLabel(
                    valence=avg("valence"), arousal=avg("arousal"),
                    playfulness=avg("playfulness"), shyness=avg("shyness"),
                    affection=avg("affection"),
                    mode=str(first.get("mode", "REACTING")),
                    source=str(first.get("source", "eval")),
                    label_quality=str(first.get("label_quality", "human")),
                )
            ))
            self.notes.append(str(first.get("notes", "")))

    def _load_old_format(self, df: pd.DataFrame):
        """Old hand-curated format: one row per sentence, 'text' column."""
        def _f(v):
            try: return float(v)
            except: return None

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            self.samples.append(EmotionSample(
                text=text,
                label=EmotionLabel(
                    valence=     _f(row.get("valence")),
                    arousal=     _f(row.get("arousal")),
                    playfulness= _f(row.get("playfulness")),
                    shyness=     _f(row.get("shyness")),
                    affection=   _f(row.get("affection")),
                    mode=str(row.get("mode", "SPEAKING")),
                    source="eval",
                    label_quality="human",
                )
            ))
            self.notes.append(str(row.get("notes", "")))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ---------------------------------------------------------------------------
# Eval set template generator
# ---------------------------------------------------------------------------

EVAL_TEMPLATE_ROWS = [
    ("hehe, no way",                        0.75, 0.60, 0.90, 0.20, 0.30, "SPEAKING", "playful denial"),
    ("uh... thanks",                        0.60, 0.30, 0.10, 0.70, 0.60, "REACTING", "shy gratitude"),
    ("wait WHAT??",                         0.50, 0.95, 0.10, 0.10, 0.00, "REACTING", "pure surprise"),
    ("that's fine, really",                 0.30, 0.20, 0.10, 0.20, 0.20, "SPEAKING", "suppressed negative"),
    ("you actually did that for me?",       0.80, 0.60, 0.20, 0.40, 0.85, "REACTING", "touched, rising affection"),
    ("...oh",                               0.20, 0.15, 0.00, 0.30, 0.10, "REACTING", "quiet realization"),
    ("stoooop you're embarrassing me",      0.70, 0.65, 0.50, 0.80, 0.50, "SPEAKING", "flustered, playful"),
    ("I knew it!! I knew it!!",             0.85, 0.90, 0.70, 0.05, 0.10, "SPEAKING", "excited vindication"),
    ("oh. okay.",                           0.25, 0.15, 0.05, 0.10, 0.10, "REACTING", "quiet disappointment"),
    ("noooo that's so cute!!",              0.90, 0.80, 0.70, 0.20, 0.60, "REACTING", "delighted"),
    ("hmm... I'm not sure",                 0.45, 0.35, 0.10, 0.30, 0.10, "THINKING", "uncertain, mild"),
    ("whatever, I don't care",              0.30, 0.25, 0.15, 0.05, 0.05, "SPEAKING", "dismissive"),
    ("are you... serious right now",        0.35, 0.60, 0.20, 0.10, 0.10, "REACTING", "disbelief, mild annoyance"),
    ("ehehe, maybe~",                       0.80, 0.55, 0.85, 0.40, 0.30, "SPEAKING", "coy, teasing"),
    ("I was so scared but also... excited?",0.60, 0.75, 0.30, 0.40, 0.20, "SPEAKING", "mixed, complex"),
]


def generate_eval_template(output_path: str = "data/eval_set.csv"):
    """Run once to scaffold the hand-curated eval set. Fill in scores manually."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "valence", "arousal", "playfulness",
                         "shyness", "affection", "mode", "notes"])
        for row in EVAL_TEMPLATE_ROWS:
            writer.writerow(row)
    print(f"Eval template written to {output_path}")
    print("Adjust scores to match your intuition, then run: build_dataset.py --eval-set data/eval_set.csv")


if __name__ == "__main__":
    generate_eval_template()
