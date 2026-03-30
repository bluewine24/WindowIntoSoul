"""
Text2Emotion model.

Architecture:
  text + mode
  -> RoBERTa-base  (frozen initially)
  -> clause attention pooling
  -> 2-layer GRU  (hidden=256)
  -> interpretable head  [M x 5]  sigmoid
  -> latent head         [M x 8]  tanh
  -> output              [M x 13]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from typing import List, Optional, Tuple
from dataclasses import dataclass

from models.segmenter import BaseSegmenter, PunctuationSegmenter, Clause


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERPRETABLE_DIMS = ["valence", "arousal", "playfulness", "shyness", "affection"]
MODES = ["SPEAKING", "LISTENING", "REACTING", "THINKING", "IDLE"]
MODE2IDX = {m: i for i, m in enumerate(MODES)}

LATENT_DIMS = 8
TOTAL_DIMS = len(INTERPRETABLE_DIMS) + LATENT_DIMS   # 13


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EmotionTrajectory:
    """Output of the model for one input."""
    interpretable: torch.Tensor     # [M x 5]  — named dims, [0,1]
    latent: torch.Tensor            # [M x 8]  — free dims, [-1,1]
    clause_texts: List[str]         # for visualization / debugging

    @property
    def full(self) -> torch.Tensor:
        """[M x 13] concatenated."""
        return torch.cat([self.interpretable, self.latent], dim=-1)

    def to_dict(self) -> dict:
        out = {}
        for i, name in enumerate(INTERPRETABLE_DIMS):
            out[name] = self.interpretable[:, i].tolist()
        out["latent"] = self.latent.tolist()
        out["clauses"] = self.clause_texts
        return out


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Learned attention pooling over token embeddings within a clause.
    Preserves emotionally salient tokens (e.g. 'totally' in 'I'm totally fine').
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, token_embeddings: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: [B, T, H]
            attention_mask:   [B, T]  — 1 for real tokens, 0 for padding
        Returns:
            pooled: [B, H]
        """
        scores = self.attn(token_embeddings).squeeze(-1)          # [B, T]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)     # [B, T, 1]
        pooled = (token_embeddings * weights).sum(dim=1)          # [B, H]
        return pooled


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Text2EmotionModel(nn.Module):

    def __init__(self,
                 encoder_name: str = "roberta-base",
                 mode_dim: int = 32,
                 gru_hidden: int = 256,
                 gru_layers: int = 2,
                 gru_dropout: float = 0.2,
                 interpretable_dims: int = 5,
                 latent_dims: int = 8,
                 freeze_encoder: bool = True,
                 segmenter: Optional[BaseSegmenter] = None):
        super().__init__()

        # --- Encoder ---
        self.encoder = RobertaModel.from_pretrained(encoder_name)
        self.encoder_hidden = self.encoder.config.hidden_size   # 768

        if freeze_encoder:
            self._freeze_encoder()

        # --- Segmenter ---
        self.segmenter = segmenter or PunctuationSegmenter()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(encoder_name)

        # --- Attention pooling (per clause) ---
        self.attn_pool = AttentionPooling(self.encoder_hidden)

        # --- Mode embedding ---
        self.mode_embedding = nn.Embedding(len(MODES), mode_dim)
        gru_input_dim = self.encoder_hidden + mode_dim

        # --- Temporal GRU ---
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        # --- Output heads ---
        self.interpretable_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden // 2, interpretable_dims),
            nn.Sigmoid(),   # [0, 1]
        )
        self.latent_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden // 2, latent_dims),
            nn.Tanh(),      # [-1, 1]
        )

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self,
                texts: List[str],
                modes: List[str],
                max_clauses: int = 16,
                max_tokens: int = 64) -> List[EmotionTrajectory]:
        """
        Args:
            texts:  list of raw input strings  (batch)
            modes:  list of mode strings per sample  e.g. ["SPEAKING", ...]
            max_clauses: clip long inputs
            max_tokens:  max tokens per clause encoding
        Returns:
            list of EmotionTrajectory, one per sample
        """
        device = next(self.parameters()).device

        # --- Mode indices ---
        mode_ids = torch.tensor(
            [MODE2IDX.get(m, 0) for m in modes], device=device
        )   # [B]

        outputs = []
        for i, (text, mode_id) in enumerate(zip(texts, mode_ids)):

            # 1. Segment into clauses
            clauses: List[Clause] = self.segmenter.segment(text)
            clauses = clauses[:max_clauses]
            if not clauses:
                clauses = [type('C', (), {'text': text})()]   # fallback

            clause_texts = [c.text for c in clauses]
            M = len(clause_texts)

            # 2. Encode each clause with RoBERTa + attention pool
            clause_embeddings = self._encode_clauses(
                clause_texts, max_tokens, device
            )   # [M, 768]

            # 3. Inject mode embedding (same mode for all clauses in sample)
            mode_emb = self.mode_embedding(mode_id.unsqueeze(0))    # [1, mode_dim]
            mode_emb = mode_emb.expand(M, -1)                       # [M, mode_dim]
            gru_input = torch.cat([clause_embeddings, mode_emb], dim=-1)  # [M, 768+mode_dim]

            # 4. GRU over clause sequence
            gru_input = gru_input.unsqueeze(0)          # [1, M, D]
            gru_out, _ = self.gru(gru_input)            # [1, M, hidden]
            gru_out = gru_out.squeeze(0)                # [M, hidden]

            # 5. Project to emotion dims
            interp = self.interpretable_head(gru_out)   # [M, 5]
            latent = self.latent_head(gru_out)          # [M, 8]

            outputs.append(EmotionTrajectory(
                interpretable=interp,
                latent=latent,
                clause_texts=clause_texts,
            ))

        return outputs

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _encode_clauses(self, clause_texts: List[str],
                        max_tokens: int, device: torch.device) -> torch.Tensor:
        """Encode all clauses in one batched forward pass through RoBERTa."""
        enc = self.tokenizer(
            clause_texts,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)

        with torch.set_grad_enabled(self.training and not self._encoder_frozen()):
            roberta_out = self.encoder(**enc)

        token_embs = roberta_out.last_hidden_state       # [M, T, 768]
        pooled = self.attn_pool(token_embs, enc.attention_mask)  # [M, 768]
        return pooled

    def _encoder_frozen(self) -> bool:
        return not next(self.encoder.parameters()).requires_grad

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, n: int = 2):
        """Unfreeze top N transformer layers. Call after phase 1 plateau."""
        # RoBERTa has 12 layers indexed 0-11
        for layer in self.encoder.encoder.layer[-(n):]:
            for param in layer.parameters():
                param.requires_grad = True
        # Also unfreeze pooler
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True
        print(f"[Model] Unfroze top {n} RoBERTa layers.")

    def param_groups(self, base_lr: float, encoder_lr: float) -> list:
        """Return parameter groups for optimizer — lower lr for encoder."""
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        other_params = (
            list(self.attn_pool.parameters()) +
            list(self.mode_embedding.parameters()) +
            list(self.gru.parameters()) +
            list(self.interpretable_head.parameters()) +
            list(self.latent_head.parameters())
        )
        groups = [{"params": other_params, "lr": base_lr}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        return groups
