#!/usr/bin/env python
"""
Standard Activation Probe Trainer:
Trains a standard linear / logistic regression probe on LLM residual stream activations.
Evaluates binary classification metrics (ROC-AUC, Accuracy, F1, Balanced Accuracy) via Stratified K-Fold.
"""

import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_shards
from src.standard_probe import StandardActivationProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Standard Activation Probe Training")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing feat.shard*.npz and meta.shard*.jsonl files",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic",
        choices=["logistic", "ridge", "mlp"],
        help="Classifier type (default: logistic)",
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
        help="Anchor position to train probe on (default: post_gen; e.g. pre_gen, first_gen, mean_gen)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Filter to a specific round index t (default: None = all rounds pooled)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer index to probe (default: None = first/only layer)",
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
    print("=" * 60)
    print("Standard Activation Probe Training")
    print("=" * 60)
    print(f"Data Directory:   {args.data_dir}")
    print(f"Classifier:       {args.classifier.upper()} (C={args.C})")
    print(f"Anchor:           {args.anchor}")
    print(f"Round Filter:     {args.round if args.round is not None else 'All Rounds'}")
    print(f"Layer:            {args.layer if args.layer is not None else 'Default'}")
    print("=" * 60)

    # 1. Load Shard Dataset
    print("\n[1/2] Loading activation shards...")
    dataset = load_shards(args.data_dir)
    print(f"  -> Loaded {dataset.num_samples} total turn records.")
    print(f"  -> Hidden Dimension: {dataset.hidden_dim}, Layers: {dataset.layer_ids}")

    # 2. Extract 2D Slice
    X, y, groups = dataset.get_slice(
        anchor=args.anchor,
        round_idx=args.round,
        layer_idx=args.layer,
    )
    print(f"  -> Sliced matrix X shape: {X.shape} (Positives: {np.sum(y == 1)}, Negatives: {np.sum(y == 0)})")

    # 3. Fit & Cross-Validate Probe
    print(f"\n[2/2] Training {args.classifier} probe ({args.n_splits}-Fold Stratified CV)...")
    probe = StandardActivationProbe(classifier_type=args.classifier, C=args.C)
    metrics, oof_scores = probe.evaluate_cv(X, y, n_splits=args.n_splits)

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
