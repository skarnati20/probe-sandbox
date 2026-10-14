#!/usr/bin/env python
"""
Standard Activation Probe Trainer:
Trains a standard linear / logistic regression probe on LLM residual stream activations.
Evaluates binary classification metrics (ROC-AUC, Accuracy, F1, Balanced Accuracy) via 5-Fold CV.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import extract_episodes, load_shards
from src.standard_probe import StandardProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Standard Activation Probe Training")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing feat.shard*.npz and meta.shard*.jsonl files",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength C (default: 1.0)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="post_gen",
        help="Anchor position to train probe on (default: post_gen)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Specific layer index to probe (default: 20)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir)
    feat_path = data_path / "features" if (data_path / "features").exists() else data_path

    print("=" * 60)
    print("Standard Activation Probe Training")
    print("=" * 60)
    print(f"Data Directory:   {data_path}")
    print(f"Anchor:           {args.anchor}")
    print(f"Layer:            {args.layer}")
    print(f"C:                {args.C}")
    print("=" * 60)

    # 1. Load Shards & Extract Episodes
    print("\n[1/2] Loading activation shards...")
    dataset = load_shards(feat_path)
    episodes = extract_episodes(dataset, target_layers=[args.layer])
    print(f"  -> Loaded {len(episodes)} episodes across {dataset.hidden_dim} dimensions.")

    # 2. Train and Evaluate CV
    print(f"\n[2/2] Training StandardProbe ({args.n_splits}-Fold CV)...")
    probe = StandardProbe(layer=args.layer, anchor=args.anchor, C=args.C)
    metrics, cv_auc = probe.evaluate_cv(episodes, n_splits=args.n_splits)

    print("\n" + "=" * 60)
    print("PROBE EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROC-AUC:            {metrics.roc_auc:.4f}")
    print(f"Accuracy:           {metrics.accuracy * 100:.2f}%")
    print(f"Balanced Accuracy:  {metrics.balanced_accuracy * 100:.2f}%")
    print(f"Precision:          {metrics.precision:.4f}")
    print(f"Recall:             {metrics.recall:.4f}")
    print(f"F1 Score:           {metrics.f1:.4f}")
    print(f"Total Samples:      {metrics.num_samples}")
    print("=" * 60)


if __name__ == "__main__":
    main()
