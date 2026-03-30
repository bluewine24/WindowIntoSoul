"""
Clause segmentation module.
Designed to be swappable — v1 uses punctuation heuristics.
Later versions can use learned boundaries or fixed bins.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Clause:
    text: str
    start_token: int    # index into original token list
    end_token: int
    raw_start: int      # char offset in original string
    raw_end: int


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, text: str) -> List[Clause]:
        """Split text into clauses. Returns list of Clause objects."""
        ...


class PunctuationSegmenter(BaseSegmenter):
    """
    V1: punctuation + conjunction heuristics.
    Handles chat-style text: ellipsis, repeated punctuation, no punctuation.
    """

    # Clause boundary signals
    HARD_BREAKS = re.compile(r'(?<=[.!?])(?:\s+|$)')
    SOFT_BREAKS = re.compile(r'(?<=,)\s+')
    ELLIPSIS    = re.compile(r'\.{2,}\s*')
    CONJUNCTION = re.compile(r'\s+(?:but|and|so|because|though|although|however|yet|while)\s+', re.IGNORECASE)

    # Emotional punctuation — keep as separate clauses
    EMOTIONAL_PUNCT = re.compile(r'([!?]{2,}|\.{3,})')

    def segment(self, text: str) -> List[Clause]:
        text = text.strip()
        if not text:
            return []

        spans = self._find_spans(text)
        clauses = []
        token_offset = 0

        for raw_start, raw_end in spans:
            chunk = text[raw_start:raw_end].strip()
            if not chunk:
                continue
            # rough token count estimate (will be overridden by tokenizer)
            tok_len = len(chunk.split())
            clauses.append(Clause(
                text=chunk,
                start_token=token_offset,
                end_token=token_offset + tok_len,
                raw_start=raw_start,
                raw_end=raw_end,
            ))
            token_offset += tok_len

        return clauses if clauses else [Clause(text=text, start_token=0,
                                               end_token=len(text.split()),
                                               raw_start=0, raw_end=len(text))]

    def _find_spans(self, text: str):
        """Find (start, end) char spans for each clause."""
        boundaries = {0, len(text)}

        for pattern in [self.HARD_BREAKS, self.ELLIPSIS, self.CONJUNCTION, self.SOFT_BREAKS]:
            for m in pattern.finditer(text):
                boundaries.add(m.start())
                boundaries.add(m.end())

        # Emotional punctuation gets its own clause
        for m in self.EMOTIONAL_PUNCT.finditer(text):
            boundaries.add(m.start())
            boundaries.add(m.end())

        boundaries = sorted(boundaries)
        return [(boundaries[i], boundaries[i+1])
                for i in range(len(boundaries) - 1)
                if boundaries[i] < boundaries[i+1]]


class FixedBinSegmenter(BaseSegmenter):
    """
    Alternative: split into fixed N-token bins.
    Useful when punctuation is absent or unreliable.
    """
    def __init__(self, bin_size: int = 8):
        self.bin_size = bin_size

    def segment(self, text: str) -> List[Clause]:
        words = text.split()
        clauses = []
        for i in range(0, len(words), self.bin_size):
            chunk = ' '.join(words[i:i + self.bin_size])
            clauses.append(Clause(
                text=chunk,
                start_token=i,
                end_token=min(i + self.bin_size, len(words)),
                raw_start=0,  # approximate
                raw_end=len(chunk),
            ))
        return clauses


def get_segmenter(name: str = "punctuation", **kwargs) -> BaseSegmenter:
    """Factory — swap segmenter here without touching model code."""
    registry = {
        "punctuation": PunctuationSegmenter,
        "fixed_bin":   FixedBinSegmenter,
    }
    if name not in registry:
        raise ValueError(f"Unknown segmenter '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
