# Text-to-Emotion Trajectory Model

A dialogue-aware emotion modeling system that produces a continuous
emotional state (`z_emotion`) from text, with temporal consistency and
character conditioning. Designed for downstream control of face/rig
animation.

---

## Overview

This model maps conversational input into a **latent emotional
trajectory** rather than isolated labels.

Key properties:

- maintains emotional state across turns\
- conditions on character/persona\
- uses dialogue context (who said what, when)\
- produces a compact latent (`z`) for animation control\
- uses interpretable supervision during training (VAD, appraisal,
  discrete)\
- applies a learned inertia gate at inference for stable expression

---

## Architecture

### 1. Utterance Encoding

RoBERTa-base (frozen)\
→ attention pooling → text_embedding (768)

text_embedding ⊕ role_embedding\
→ Linear → utterance_embedding (768)

---

### 2. Character Conditioning

character_vector\
→ MLP → h_0 (initial GRU hidden state)\
→ MLP → char_step (appended at every timestep)\
→ prepended to context_buffer

---

### 3. Context Buffer (Memory)

memory_t = \[\
utterance_embedding ;\
role_embedding ;\
z_stable\* ;\
turn_distance_embedding\
\]

- sliding window over last N turns\

- includes both semantic and emotional history

- uses:

- `z_emotion` during training\

- `z_stable` during inference

---

### 4. Recurrent Core

cross_attention(prev_h, context_buffer) → context_vector

concat(utterance_embedding, context_vector, char_step)\
→ GRU (2 layers, hidden=512)\
→ h_t

h_t\
→ Linear → LayerNorm → Tanh\
→ z_emotion (128)

---

### 5. Interpretable Heads (Training Only)

z_emotion → VAD (3) → Tanh\
z_emotion → appraisal (5) → Tanh\
z_emotion → discrete (14) → Softmax (auxiliary)

---

## Training Objectives

### Losses (Training Only)

- smoothness loss (label-aware, light weight)\
- contrastive loss (discrete label + role, soft margin)\
- consistency loss (same label + role → nearby latent, low weight)

---

## Inertia Gate (Temporal Stabilization)

### v1 (simple)

gate = sigmoid(W \* \|z_t - z_prev\| + b)

### v2 (context-aware)

gate = sigmoid(MLP(\[z_prev ; z_t ; \|z_t - z_prev\| ;
context_vector\]))

### Final update

z_stable = gate \* z_t + (1 - gate) \* z_prev

---

## Output

z_stable → face / rig decoder

---

## Training Procedure

### Stage 1 --- Base Model

Train GRU, projection, and heads with all losses. RoBERTa frozen.

### Stage 2 --- Inertia Gate

Freeze base. Train gate module.

### Stage 3 --- Joint Fine-Tuning

Unfreeze all. Train with small learning rate.

---

## Summary

Emotion is modeled as a continuous, context-aware trajectory:

emotion = f(text, role, character, memory, time)
