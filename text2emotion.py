from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from datasets import APPRAISAL_NAMES, DISCRETE_EMOTIONS, DialogueExample

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, sequence_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.projection(sequence_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(sequence_embeddings * weights, dim=1)


class SimpleWhitespaceTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.cls_id = 1
        self.sep_id = 2
        self._pattern = re.compile(r"\w+|[^\w\s]")

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        hashed = int(digest, 16) % max(self.vocab_size - 3, 1)
        return hashed + 3

    def _encode_text(self, text: str, max_length: int) -> list[int]:
        pieces = self._pattern.findall(text.lower())
        token_ids = [self.cls_id]
        for token in pieces[: max(0, max_length - 2)]:
            token_ids.append(self._token_to_id(token))
        token_ids.append(self.sep_id)
        return token_ids[:max_length]

    def batch_encode(self, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
        sequences = [self._encode_text(text, max_length=max_length) for text in texts]
        width = max(len(sequence) for sequence in sequences)
        input_ids = []
        masks = []
        for sequence in sequences:
            pad = width - len(sequence)
            input_ids.append(sequence + [self.pad_id] * pad)
            masks.append([1] * len(sequence) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        }


class SimpleTextEncoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.tokenizer = SimpleWhitespaceTokenizer(vocab_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pooling = AttentionPooling(hidden_size)

    def forward(self, texts: list[str], max_tokens: int, device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer.batch_encode(texts, max_length=max_tokens, device=device)
        embeddings = self.embedding(encoded["input_ids"])
        contextualized, _ = self.encoder(embeddings)
        return self.pooling(contextualized, encoded["attention_mask"])


class TextBackbone(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        hidden_size: int,
        max_tokens: int,
        freeze_text_encoder: bool,
        allow_mock_encoder_fallback: bool,
        mock_vocab_size: int,
        local_files_only: bool,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.mode = "simple"
        self.hidden_size = hidden_size
        self._encoder = SimpleTextEncoder(hidden_size=hidden_size, vocab_size=mock_vocab_size)
        self._tokenizer = None
        self._pooling = None

        if AutoModel is not None and AutoTokenizer is not None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(encoder_name, local_files_only=local_files_only, use_fast=True)
                self._encoder = AutoModel.from_pretrained(encoder_name, local_files_only=local_files_only)
                self._pooling = AttentionPooling(self._encoder.config.hidden_size)
                self.hidden_size = self._encoder.config.hidden_size
                self.mode = "hf"
            except Exception as exc:
                if not allow_mock_encoder_fallback:
                    raise
                warnings.warn(
                    f"Falling back to the built-in mock text encoder because `{encoder_name}` "
                    f"could not be loaded locally: {exc}"
                )

        if self.mode == "hf" and freeze_text_encoder:
            self.set_encoder_trainable(False)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self._encoder.parameters():
            parameter.requires_grad = trainable

    def pretrained_parameters(self) -> list[nn.Parameter]:
        if self.mode == "hf":
            return list(self._encoder.parameters())
        return []

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if self.mode == "hf":
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = self._encoder(**encoded)
            return self._pooling(outputs.last_hidden_state, encoded["attention_mask"])
        return self._encoder(texts=texts, max_tokens=self.max_tokens, device=device)


class ContextAwareInertiaGate(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, gate_type: str, hidden_dim: int):
        super().__init__()
        self.gate_type = gate_type
        if gate_type == "simple":
            self.linear = nn.Linear(1, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear((latent_dim * 3) + context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, previous_latent: torch.Tensor, current_latent: torch.Tensor, context_vector: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(current_latent - previous_latent)
        if self.gate_type == "simple":
            logits = self.linear(torch.norm(delta, dim=-1, keepdim=True))
        else:
            features = torch.cat([previous_latent, current_latent, delta, context_vector], dim=-1)
            logits = self.mlp(features)
        return torch.sigmoid(logits)


@dataclass
class DialogueEmotionOutput:
    z_emotion: torch.Tensor
    z_stable: torch.Tensor
    vad: torch.Tensor
    appraisal: torch.Tensor
    discrete_logits: torch.Tensor
    gates: torch.Tensor
    attention_weights: list[torch.Tensor]

    def to_dict(self, dialogue: DialogueExample) -> dict[str, Any]:
        discrete_probs = torch.softmax(self.discrete_logits, dim=-1)
        predicted_ids = torch.argmax(discrete_probs, dim=-1)
        turns = []
        for index, turn in enumerate(dialogue.turns):
            turns.append(
                {
                    "index": index,
                    "role": turn.role,
                    "text": turn.text,
                    "gate": float(self.gates[index].item()),
                    "predicted_discrete_id": int(predicted_ids[index].item()),
                    "predicted_discrete_name": DISCRETE_EMOTIONS[int(predicted_ids[index].item())],
                    "vad": [round(value, 4) for value in self.vad[index].tolist()],
                    "appraisal": {
                        name: round(value, 4)
                        for name, value in zip(APPRAISAL_NAMES, self.appraisal[index].tolist())
                    },
                    "z_emotion": [round(value, 4) for value in self.z_emotion[index].tolist()],
                    "z_stable": [round(value, 4) for value in self.z_stable[index].tolist()],
                }
            )
        return {"dialogue_id": dialogue.dialogue_id, "character_id": dialogue.character_id, "turns": turns}


class TextToEmotionTrajectoryModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_cfg = config["model"]
        self.config = config
        self.freeze_text_encoder = bool(model_cfg["freeze_text_encoder"])
        self.role_vocab = [role.lower() for role in model_cfg["role_vocab"]]
        self.role_to_id = {role: index for index, role in enumerate(self.role_vocab)}
        self.turn_distance_buckets = int(model_cfg["turn_distance_buckets"])
        self.memory_window = int(model_cfg["memory_window"])
        self.recurrent_hidden_size = int(model_cfg["recurrent_hidden_size"])
        self.recurrent_layers = int(model_cfg["recurrent_layers"])
        self.latent_dim = int(model_cfg["latent_dim"])

        self.text_backbone = TextBackbone(
            encoder_name=model_cfg["encoder_name"],
            hidden_size=int(model_cfg["encoder_hidden_size"]),
            max_tokens=int(model_cfg["max_tokens"]),
            freeze_text_encoder=self.freeze_text_encoder,
            allow_mock_encoder_fallback=bool(model_cfg["allow_mock_encoder_fallback"]),
            mock_vocab_size=int(model_cfg["mock_vocab_size"]),
            local_files_only=bool(model_cfg["local_files_only"]),
        )

        role_embedding_dim = int(model_cfg["role_embedding_dim"])
        utterance_dim = int(model_cfg["utterance_dim"])
        character_dim = int(model_cfg["character_dim"])
        character_hidden_dim = int(model_cfg["character_hidden_dim"])
        char_step_dim = int(model_cfg["char_step_dim"])
        turn_distance_embedding_dim = int(model_cfg["turn_distance_embedding_dim"])

        self.role_embedding = nn.Embedding(len(self.role_vocab), role_embedding_dim)
        self.utterance_projection = nn.Linear(self.text_backbone.hidden_size + role_embedding_dim, utterance_dim)
        self.character_to_h0 = _mlp(character_dim, character_hidden_dim, self.recurrent_layers * self.recurrent_hidden_size)
        self.character_to_step = _mlp(character_dim, character_hidden_dim, char_step_dim)
        self.turn_distance_embedding = nn.Embedding(self.turn_distance_buckets, turn_distance_embedding_dim)

        self.memory_dim = utterance_dim + role_embedding_dim + self.latent_dim + turn_distance_embedding_dim
        self.character_memory_projection = nn.Sequential(
            nn.Linear(char_step_dim, self.memory_dim),
            nn.Tanh(),
        )

        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.recurrent_hidden_size,
            num_heads=int(model_cfg["context_attention_heads"]),
            batch_first=True,
            kdim=self.memory_dim,
            vdim=self.memory_dim,
        )
        self.recurrent_core = nn.GRU(
            input_size=utterance_dim + self.recurrent_hidden_size + char_step_dim,
            hidden_size=self.recurrent_hidden_size,
            num_layers=self.recurrent_layers,
            batch_first=True,
            dropout=float(model_cfg["recurrent_dropout"]) if self.recurrent_layers > 1 else 0.0,
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(self.recurrent_hidden_size, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.Tanh(),
        )

        self.vad_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, int(model_cfg["vad_dim"])),
            nn.Tanh(),
        )
        self.appraisal_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, int(model_cfg["appraisal_dim"])),
            nn.Tanh(),
        )
        self.discrete_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, int(model_cfg["discrete_dim"])),
        )
        self.inertia_gate = ContextAwareInertiaGate(
            latent_dim=self.latent_dim,
            context_dim=self.recurrent_hidden_size,
            gate_type=str(model_cfg["gate_type"]),
            hidden_dim=int(model_cfg["gate_hidden_dim"]),
        )

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def set_stage(self, stage: str) -> None:
        if stage not in {"base", "gate", "joint"}:
            raise ValueError(f"Unknown training stage: {stage}")

        self._set_module_trainable(self, False)
        if stage == "base":
            base_modules = [
                self.role_embedding,
                self.utterance_projection,
                self.character_to_h0,
                self.character_to_step,
                self.turn_distance_embedding,
                self.character_memory_projection,
                self.context_attention,
                self.recurrent_core,
                self.latent_projection,
                self.vad_head,
                self.appraisal_head,
                self.discrete_head,
            ]
            for module in base_modules:
                self._set_module_trainable(module, True)
            text_encoder_trainable = (self.text_backbone.mode != "hf") or (not self.freeze_text_encoder)
            self.text_backbone.set_encoder_trainable(text_encoder_trainable)
        elif stage == "gate":
            self._set_module_trainable(self.inertia_gate, True)
        else:
            self._set_module_trainable(self, True)

    def parameter_groups(self, base_lr: float, encoder_lr: Optional[float] = None) -> list[dict[str, Any]]:
        encoder_ids = {id(parameter) for parameter in self.text_backbone.pretrained_parameters() if parameter.requires_grad}
        base_params = []
        encoder_params = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in encoder_ids and encoder_lr is not None:
                encoder_params.append(parameter)
            else:
                base_params.append(parameter)
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": base_lr})
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        return groups

    def role_id(self, role: str) -> int:
        return self.role_to_id.get(role.lower(), self.role_to_id["other"])

    def heads_from_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vad_head(latent), self.appraisal_head(latent), self.discrete_head(latent)

    def forward(
        self,
        dialogues: list[DialogueExample],
        use_stable_history: Optional[bool] = None,
    ) -> list[DialogueEmotionOutput]:
        device = next(self.parameters()).device
        history_mode = (not self.training) if use_stable_history is None else use_stable_history
        outputs: list[DialogueEmotionOutput] = []

        for dialogue in dialogues:
            texts = [turn.text for turn in dialogue.turns]
            role_ids = torch.tensor([self.role_id(turn.role) for turn in dialogue.turns], dtype=torch.long, device=device)
            distance_ids = torch.tensor(
                [max(0, min(self.turn_distance_buckets - 1, turn.turn_distance)) for turn in dialogue.turns],
                dtype=torch.long,
                device=device,
            )

            text_embeddings = self.text_backbone(texts=texts, device=device)
            role_embeddings = self.role_embedding(role_ids)
            utterance_embeddings = self.utterance_projection(torch.cat([text_embeddings, role_embeddings], dim=-1))

            character_vector = torch.tensor(dialogue.character_vector, dtype=torch.float32, device=device)
            hidden_state = self.character_to_h0(character_vector).view(self.recurrent_layers, 1, self.recurrent_hidden_size)
            char_step = self.character_to_step(character_vector)
            char_memory = self.character_memory_projection(char_step).unsqueeze(0)

            latent_steps = []
            stable_steps = []
            vad_steps = []
            appraisal_steps = []
            discrete_steps = []
            gate_steps = []
            attention_steps = []
            memory_items: list[torch.Tensor] = []
            previous_stable: Optional[torch.Tensor] = None

            for turn_index in range(len(dialogue.turns)):
                if memory_items:
                    history = torch.stack(memory_items[-self.memory_window :], dim=0)
                    context_buffer = torch.cat([char_memory, history], dim=0)
                else:
                    context_buffer = char_memory

                query = hidden_state[-1].unsqueeze(1)
                context_vector, attention_weights = self.context_attention(
                    query=query,
                    key=context_buffer.unsqueeze(0),
                    value=context_buffer.unsqueeze(0),
                    need_weights=True,
                )
                context_vector = context_vector.squeeze(0).squeeze(0)
                attention_steps.append(attention_weights.squeeze(0).squeeze(0).detach().cpu())

                step_input = torch.cat([utterance_embeddings[turn_index], context_vector, char_step], dim=-1)
                step_output, hidden_state = self.recurrent_core(step_input.view(1, 1, -1), hidden_state)
                z_emotion = self.latent_projection(step_output.squeeze(0).squeeze(0))

                if previous_stable is None:
                    gate = z_emotion.new_tensor([1.0])
                    z_stable = z_emotion
                else:
                    gate = self.inertia_gate(previous_stable, z_emotion, context_vector)
                    z_stable = gate * z_emotion + (1.0 - gate) * previous_stable

                vad, appraisal, discrete_logits = self.heads_from_latent(z_emotion)
                history_latent = z_stable if history_mode else z_emotion
                memory_item = torch.cat(
                    [
                        utterance_embeddings[turn_index],
                        role_embeddings[turn_index],
                        history_latent,
                        self.turn_distance_embedding(distance_ids[turn_index]),
                    ],
                    dim=-1,
                )
                memory_items.append(memory_item)
                previous_stable = z_stable

                latent_steps.append(z_emotion)
                stable_steps.append(z_stable)
                vad_steps.append(vad)
                appraisal_steps.append(appraisal)
                discrete_steps.append(discrete_logits)
                gate_steps.append(gate.view(1))

            outputs.append(
                DialogueEmotionOutput(
                    z_emotion=torch.stack(latent_steps, dim=0),
                    z_stable=torch.stack(stable_steps, dim=0),
                    vad=torch.stack(vad_steps, dim=0),
                    appraisal=torch.stack(appraisal_steps, dim=0),
                    discrete_logits=torch.stack(discrete_steps, dim=0),
                    gates=torch.stack(gate_steps, dim=0),
                    attention_weights=attention_steps,
                )
            )

        return outputs
