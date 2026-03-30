"""
inference.py — interactive testing of the text2emotion model.

Usage:
    python inference.py                        # interactive REPL
    python inference.py --checkpoint path.pt   # load specific checkpoint
    python inference.py --eval                 # run on full eval set
"""

import argparse
import torch

from models.text2emotion import Text2EmotionModel, MODES
from evaluation.visualizer import TrajectoryVisualizer
from data.datasets import EvalSetDataset


# ---------------------------------------------------------------------------
# Sanity check sentences — run these first after any training milestone
# ---------------------------------------------------------------------------

SANITY_CHECKS = [
    ("hehe, no way",                        "SPEAKING"),
    ("uh... thanks",                        "REACTING"),
    ("wait WHAT??",                         "REACTING"),
    ("that's fine, really",                 "SPEAKING"),
    ("you actually did that for me?",       "REACTING"),
    ("...oh",                               "REACTING"),
    ("stoooop you're embarrassing me",      "SPEAKING"),
    ("I knew it!! I knew it!!",             "SPEAKING"),
    ("oh. okay.",                           "REACTING"),
    ("noooo that's so cute!!",              "REACTING"),
    ("hmm... I'm not sure about this",      "THINKING"),
    ("ehehe, maybe~",                       "SPEAKING"),
]


def load_model(checkpoint_path: str = None,
               config_path: str = "configs/config.yaml") -> Text2EmotionModel:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    model = Text2EmotionModel(
        encoder_name=       mc["encoder"],
        mode_dim=           mc["mode_dim"],
        gru_hidden=         mc["gru_hidden"],
        gru_layers=         mc["gru_layers"],
        gru_dropout=        mc["gru_dropout"],
        interpretable_dims= mc["interpretable_dims"],
        latent_dims=        mc["latent_dims"],
        freeze_encoder=     False,   # inference mode
    )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {checkpoint_path} (epoch={ckpt['epoch']})")
    else:
        print("No checkpoint — using random weights (for architecture testing only)")

    model.eval()
    return model


def run_sanity_checks(model: Text2EmotionModel):
    viz = TrajectoryVisualizer()
    print("\n" + "="*60)
    print("  SANITY CHECK — inspect these trajectories")
    print("="*60)
    texts = [s[0] for s in SANITY_CHECKS]
    modes = [s[1] for s in SANITY_CHECKS]
    with torch.no_grad():
        trajectories = model(texts, modes)
    for (text, mode), traj in zip(SANITY_CHECKS, trajectories):
        viz.print_trajectory(text=text, trajectory=traj)


def interactive_repl(model: Text2EmotionModel):
    viz = TrajectoryVisualizer()
    print("\nText2Emotion REPL")
    print("Type a sentence, optionally prefix mode: SPEAKING: I'm fine")
    print("Type 'quit' to exit, 'sanity' to run sanity checks\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() == "quit":
            break
        if raw.lower() == "sanity":
            run_sanity_checks(model)
            continue

        # Parse optional mode prefix
        mode = "SPEAKING"
        text = raw
        for m in MODES:
            if raw.upper().startswith(m + ":"):
                mode = m
                text = raw[len(m)+1:].strip()
                break

        with torch.no_grad():
            traj = model([text], [mode])[0]
        viz.print_trajectory(text=text, trajectory=traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--sanity",     action="store_true", help="Run sanity checks and exit")
    parser.add_argument("--eval",       action="store_true", help="Run on eval set")
    parser.add_argument("--eval-path",  type=str, default="data/eval_set.csv")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config)

    if args.sanity:
        run_sanity_checks(model)
    elif args.eval:
        if not __import__("pathlib").Path(args.eval_path).exists():
            print(f"Eval set not found: {args.eval_path}")
            print("Run: python data/datasets.py  to generate template")
        else:
            eval_set = EvalSetDataset(args.eval_path)
            viz = TrajectoryVisualizer()
            with torch.no_grad():
                for sample in eval_set:
                    traj = model([sample.text], [sample.label.mode])[0]
                    viz.print_trajectory(sample.text, traj, sample.label)
    else:
        interactive_repl(model)
