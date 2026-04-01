# Text-to-Emotion Trajectory Model

A dialogue-aware PyTorch project that turns a sequence of utterances into a
continuous emotion trajectory. Instead of predicting one isolated label per
line, the model maintains a latent emotional state across turns and smooths it
with an inertia gate for downstream face or rig control.

## What Is Implemented

This repository now contains a runnable end-to-end scaffold for the design in
the original project note:

- dialogue dataset loading from JSONL
- character-conditioned recurrent emotion model
- memory buffer with cross-attention over recent turns
- interpretable heads for VAD, appraisal, and discrete emotion classes
- inertia gate that produces a stabilized latent state
- staged training loop: base -> gate -> joint
- inference CLI and trajectory visualization
- sample data generator so the project can run immediately

## System Diagram

```text
text turn
  |
  v
encoder -> text embedding + role embedding -> utterance_embedding
                                                |
                                                v
previous turns -> context memory -> cross-attention -> context_vector
                      ^                                  |
                      |                                  v
     latent_history = z_emotion (training) or z_stable (inference)

character_vector -> h0, char_step ----------------------+
                                                        |
                                                        v
                                   GRU -> z_emotion -> gate -> z_stable -> output
```

## Project Layout

```text
.
|- README.md
|- config.yaml
|- datasets.py
|- text2emotion.py
|- losses.py
|- trainer.py
|- annotate_appraisals.py
|- inference.py
|- visualizer.py
|- requirements.txt
`- sample_data/              # created by datasets.py
```

## Quick Start

Use the included virtual environment if you want to match the verified setup on
this machine.

### 1. Generate starter data

```powershell
basic_env\Scripts\python.exe datasets.py --write-sample-data
```

This creates:

- `sample_data/train.jsonl`
- `sample_data/val.jsonl`
- `sample_data/example_dialogue.json`

### 2. Run a small smoke test

```powershell
basic_env\Scripts\python.exe trainer.py --smoke-test
```

This runs one batch for each training stage and writes checkpoints to
`checkpoints/`.

### 3. Train normally

```powershell
basic_env\Scripts\python.exe trainer.py
```

### 4. Run inference

```powershell
basic_env\Scripts\python.exe inference.py --dialogue-path sample_data/example_dialogue.json --plot-path outputs/demo.png
```

This prints a per-turn summary, writes JSON output, and optionally saves a
trajectory plot.

### 5. Build a unified training corpus

```powershell
basic_env\Scripts\python.exe preprocess_datasets.py --config preprocess_recipe.yaml
```

This converts raw source datasets into the unified target-character schema,
creates one sample per target character per dialogue, normalizes available
labels, and writes dialogue-level train/val/test splits to `processed_data/`.

## Data Format

Training and validation files are JSONL. Each line is one dialogue with a
single target character and a sequence of turns.

Example:

```json
{
  "dialogue_id": "train-001",
  "character_id": "ava",
  "character_vector": [0.12, -0.03, 0.44],
  "turns": [
    {
      "role": "self",
      "text": "I cannot believe we finally shipped it.",
      "turn_distance": 0,
      "labels": {
        "vad": [0.75, 0.55, 0.30],
        "appraisal": [0.80, 0.35, 0.70, 0.60, 0.45],
        "discrete": 9
      }
    }
  ]
}
```

Fields:

- `character_vector`: fixed persona embedding for the target character
- `role`: role relative to the target character, currently `self`, `other`,
  or `narrator`
- `turn_distance`: simple recency feature bucketed by the model
- `vad`: valence, arousal, dominance
- `appraisal`: five aligned appraisal values in this order:
  `coping`, `goal_relevance`, `novelty`, `pleasantness`, `norm_fit`
- `discrete`: integer class id from the 14-class discrete label set

If a dataset does not provide one of these labels, that target stays missing
and its supervised loss is masked out. This is the current situation for
appraisal on the merged real-data corpus.

Planned extension for synthetic appraisal supervision:

```json
"labels": {
  "vad": [0.20, 0.45, -0.10],
  "appraisal": [0.30, 0.85, null, -0.25, 0.10],
  "appraisal_confidence": [0.95, 0.80, 0.00, 0.60, 0.70],
  "discrete": 4
}
```

`appraisal_confidence` is the supported schema for confidence-weighted
synthetic appraisal training.

The dataset intentionally does not store raw speaker names. The model only uses
`role` (`self` or `other`) because names alone do not provide useful emotional
signal here. If relationship modeling becomes important later, speaker identity
can come back as a learned `speaker_embedding`, not as raw text names.

For multi-speaker conversations, preprocessing creates one training sample per
target character per dialogue. The target character's own turns become
`role=self`, and all other turns become `role=other`.

## How The Model Works

The main model lives in `text2emotion.py`.

### 1. Text backbone

Each utterance is encoded independently by `TextBackbone`.

- uses RoBERTa if available
- otherwise uses a lightweight GRU encoder

This gives one dense embedding per utterance.

### 2. Role conditioning

Each turn's role is embedded and concatenated with the text embedding. That
merged vector is projected into `utterance_embedding`.

This happens in:

- `TextToEmotionTrajectoryModel.role_embedding`
- `TextToEmotionTrajectoryModel.utterance_projection`

### 3. Character conditioning

The target character embedding influences the sequence in two ways:

- `character_to_h0` initializes the recurrent hidden state
- `character_to_step` produces a per-step character feature appended to every
  timestep input

This lets the same dialogue content produce different trajectories for
different personas.

### 4. Context buffer and cross-attention

For each turn, the model builds a memory item:

```text
[utterance_embedding ; role_embedding ; latent_history ; turn_distance_embedding]

latent_history = z_emotion during training
latent_history = z_stable during inference
```

The memory buffer stores the most recent items up to `memory_window`. Before
processing the next turn, the model attends from the previous hidden state into
that memory with `nn.MultiheadAttention`.

The attended result becomes `context_vector`, which summarizes the emotionally
relevant recent history.

Unlike standard dialogue models, memory stores both semantic embeddings and
past emotional states, allowing future predictions to depend on how previous
utterances were felt, not just what was said.

### 5. Recurrent core

At each timestep the model concatenates:

```text
[utterance_embedding ; context_vector ; char_step]
```

That vector is passed into a multi-layer GRU. The GRU output is projected with:

```text
Linear -> LayerNorm -> Tanh
```

to produce the raw latent emotional state `z_emotion`.

### 6. Interpretable heads

Three heads read from `z_emotion`:

- `vad_head`: 3 values in `[-1, 1]`
- `appraisal_head`: 5 values in `[-1, 1]`
- `discrete_head`: 14-class logits

These are used as training targets and also provide human-readable diagnostics
during inference.

### 7. Inertia gate

The raw latent can change too sharply between turns, so the model also learns a
gate:

```text
gate = sigmoid(MLP([z_prev ; z_t ; |z_t - z_prev| ; context_vector]))
z_stable = gate * z_t + (1 - gate) * z_prev
```

`z_emotion` is the unconstrained current estimate.
`z_stable` is the temporally stabilized version intended for animation control.

## Training Logic

Training is split into three explicit stages in `trainer.py`.

With the current merged corpus, training should not be blocked on appraisal.
The model keeps the appraisal head so the latent stays compatible with later
supervision, but missing appraisal labels are masked and contribute zero loss.
Stage-specific appraisal weighting is controlled by
`training.appraisal_stage_weights` in `config.yaml`.

### Stage 1: base

`model.set_stage("base")`

Train the backbone around `z_emotion`:

- VAD regression
- discrete classification
- label-aware smoothness
- contrastive loss on same-label-and-role pairs
- consistency loss to keep same-label + same-role states close in latent space
- appraisal head present, but appraisal loss masked until labels exist

The inertia gate is frozen in this stage.

### Stage 2: gate

`model.set_stage("gate")`

Freeze the base model and train only the inertia gate using losses on
`z_stable`:

- VAD/discrete supervision through the stabilized latent
- smoothness pressure on `z_stable`
- fidelity loss: `||z_stable - z_emotion||` to prevent over-smoothing
- later add confidence-weighted appraisal supervision when synthetic labels are
  available

### Stage 3: joint

`model.set_stage("joint")`

Unfreeze everything and optimize both objectives together, with gate losses
scaled by `joint_gate_scale`.

Recommended order with the current dataset mix:

1. Train the base system now with discrete, VAD, and temporal losses.
2. Add synthetic appraisal labels later instead of forcing noisy supervision
   into an unstable latent too early.
3. Jointly fine-tune all heads after the appraisal pipeline is validated.

## Synthetic Appraisal Plan

Appraisal is the only major supervision gap in the current corpus. The fastest
practical path is to bootstrap it from the dialogue data you already have.

For each turn, label:

- `coping`
- `goal_relevance`
- `novelty`
- `pleasantness`
- `norm_fit`

Use as annotation inputs:

- current turn text
- recent dialogue context
- target-character-relative `role`

Recommended process:

1. Generate synthetic appraisal labels with an LLM on the training split.
2. Store both appraisal values and per-dimension confidence scores.
3. Weight appraisal loss by confidence so uncertain labels act as weak hints.
4. Manually review a few hundred samples to calibrate prompts and catch bias.

This keeps the base emotion model moving now while leaving a clean path toward
later appraisal supervision.

Example command:

```powershell
basic_env\Scripts\python.exe annotate_appraisals.py --input-path processed_data/train.jsonl --output-path processed_data/train_appraisal_annotated.jsonl --provider mock --report-path outputs/appraisal_annotation_report.json
```

Swap `--provider mock` for `--provider openai_compatible` and set
`OPENAI_API_KEY` when you want real LLM annotations.

## How The Code Flows

### `datasets.py`

- defines `DialogueExample`, `DialogueTurn`, and `TurnLabels`
- loads JSONL files into Python dataclasses
- provides `write_sample_data()` to create a runnable starter dataset

### `preprocess_datasets.py`

- converts raw single-turn or dialogue datasets into the unified schema
- maps source labels into the shared 14-class discrete space
- scales VAD into a common range and preserves sparse appraisal labels with
  null masks
- creates one sample per target character per dialogue
- splits by dialogue id to avoid train/val/test leakage

### `text2emotion.py`

- defines the full model
- uses RoBERTa if available, otherwise a lightweight GRU encoder
- builds memory, recurrent dynamics, auxiliary heads, and gate
- returns `DialogueEmotionOutput`, which contains both raw and stabilized
  latents plus readable predictions

### `losses.py`

- converts turn labels into tensors and masks
- applies confidence-weighted appraisal regression when
  `appraisal_confidence` is present
- computes regression, classification, smoothness, contrastive, and
  consistency losses
- switches behavior depending on training stage and appraisal stage weight

### `annotate_appraisals.py`

- reads unified dialogue JSONL and annotates each turn with appraisal values
- stores per-dimension `appraisal_confidence`
- supports a local `mock` annotator for dry runs and an `openai_compatible`
  backend for real synthetic labeling

### `trainer.py`

- loads config and datasets
- creates stage-specific optimizers
- runs train/validation loops
- saves `checkpoints/latest.pt`, `checkpoints/best.pt`, and
  `checkpoints/training_metrics.json`

### `inference.py`

- loads config and optional checkpoint
- loads a dialogue JSON file or uses the sample dialogue
- runs the model with stable-history mode enabled
- prints a readable summary and writes JSON output

### `visualizer.py`

- prints compact turn-by-turn summaries
- plots VAD, appraisal, latent norms, and gate values

## Config

Most project behavior is controlled from `config.yaml`.

Useful sections:

- `model`: architecture dimensions and encoder behavior
- `training`: epochs, learning rates, loss weights, checkpoint directory
- `training.appraisal_stage_weights`: stage-specific scaling for appraisal loss
- `data`: train/validation file paths
- `inference`: default JSON output path

Notable options:

- `local_files_only: true`
  Keeps Hugging Face loading offline. If the encoder is not present locally,
  the fallback encoder is used.
- `allow_mock_encoder_fallback: true`
  Lets the project run even without a downloaded transformer model.

## Outputs

After training or smoke testing, the main artifacts are:

- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- `checkpoints/training_metrics.json`
- `outputs/inference.json`
- `outputs/*.png` if plotting is enabled

## Current Limitations

This is a solid scaffold, not a finished research system yet.

- sample data is synthetic and tiny
- the merged real-data corpus currently has no direct appraisal supervision
- there is no dedicated evaluation script beyond smoke training and plotting
- the fallback text encoder is useful for offline execution, but much weaker
  than a trained transformer backbone
- the model predicts turn-level emotion states; it does not yet include a face
  decoder or blendshape head

## Recommended Next Steps

If you want to keep developing this project, the highest-value next steps are:

1. Keep training the current merged corpus with VAD, discrete, and temporal
   losses while appraisal stays masked.
2. Add synthetic appraisal labels plus per-dimension confidence scores on the
   training split.
3. Add a proper experiment/evaluation loop with held-out metrics by task.
4. Reintroduce speaker identity only as a learned `speaker_embedding` if you
   need relationship-aware modeling.
5. Add a downstream decoder from `z_stable` to facial action units or rig
   controls.
