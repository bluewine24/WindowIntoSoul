from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets import APPRAISAL_NAMES, DISCRETE_EMOTIONS, DialogueExample
from text2emotion import DialogueEmotionOutput


class TrajectoryVisualizer:
    def print_summary(self, dialogue: DialogueExample, output: DialogueEmotionOutput) -> None:
        probabilities = torch.softmax(output.discrete_logits, dim=-1)
        predicted = torch.argmax(probabilities, dim=-1)
        print(f"Dialogue: {dialogue.dialogue_id} | Character: {dialogue.character_id}")
        for index, turn in enumerate(dialogue.turns):
            print(f"[{index}] {turn.speaker} ({turn.role})")
            print(f"    text: {turn.text}")
            print(
                "    emotion:"
                f" {DISCRETE_EMOTIONS[int(predicted[index].item())]}"
                f" | gate={output.gates[index].item():.3f}"
                f" | vad={', '.join(f'{value:.2f}' for value in output.vad[index].tolist())}"
            )

    def plot_dialogue(self, dialogue: DialogueExample, output: DialogueEmotionOutput, save_path: str) -> None:
        turns = list(range(len(dialogue.turns)))
        figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

        vad = output.vad.detach().cpu()
        axes[0].plot(turns, vad[:, 0], marker="o", label="valence")
        axes[0].plot(turns, vad[:, 1], marker="o", label="arousal")
        axes[0].plot(turns, vad[:, 2], marker="o", label="dominance")
        axes[0].set_ylabel("VAD")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        appraisal = output.appraisal.detach().cpu()
        for index, name in enumerate(APPRAISAL_NAMES):
            axes[1].plot(turns, appraisal[:, index], marker="o", label=name)
        axes[1].set_ylabel("Appraisal")
        axes[1].legend(loc="upper right", ncol=2)
        axes[1].grid(True, alpha=0.3)

        stable_norm = torch.norm(output.z_stable.detach().cpu(), dim=-1)
        raw_norm = torch.norm(output.z_emotion.detach().cpu(), dim=-1)
        gates = output.gates.detach().cpu().squeeze(-1)
        axes[2].plot(turns, raw_norm, marker="o", label="|z_emotion|")
        axes[2].plot(turns, stable_norm, marker="o", label="|z_stable|")
        axes[2].plot(turns, gates, marker="o", label="gate")
        axes[2].set_ylabel("Latent / Gate")
        axes[2].set_xlabel("Turn")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        labels = [f"{turn.speaker}\n{turn.text[:36]}" for turn in dialogue.turns]
        axes[2].set_xticks(turns)
        axes[2].set_xticklabels(labels, rotation=15, ha="right")

        figure.suptitle(f"Emotion Trajectory: {dialogue.dialogue_id}")
        figure.tight_layout()
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
