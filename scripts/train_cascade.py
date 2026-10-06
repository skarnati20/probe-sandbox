#!/usr/bin/env python
"""
Recall-Controlled Probe Cascade Trainer & Evaluator:
Faithfully reproduces the methodology from 'Doomed from the Start' (2026).
1. Evaluates GroupKFold(groups=task_idx) round-wise linear probes per turn t (AUC-over-round).
2. Applies Clopper-Pearson binomial lower confidence bound calibration on positive recall.
3. Evaluates episode-level early aborts, preserved recall, and token compute savings.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cascade import RecallControlledCascade
from src.data import load_shards
from src.roundwise_probe import RoundwiseProber


def parse_args():
    parser = argparse.ArgumentParser(description="Recall-Controlled Cascade Probe Training")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing feat.shard*.npz and meta.shard*.jsonl files",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.90, 0.92, 0.95, 0.97],
        help="Recall targets to evaluate (default: 0.90 0.92 0.95 0.97)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="post_gen",
        help="Anchor position (default: post_gen)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer index to evaluate (default: None = default layer)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Clopper-Pearson confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of GroupKFold task splits (default: 5)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression inverse regularization C (default: 1.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("Recall-Controlled Probe Cascade Evaluation")
    print("=" * 65)
    print(f"Data Directory:     {args.data_dir}")
    print(f"Anchor:             {args.anchor}")
    print(f"Layer:              {args.layer if args.layer is not None else 'Default'}")
    print(f"Recall Targets:     {args.targets}")
    print(f"Confidence Level:   {args.confidence * 100:.1f}% (Clopper-Pearson)")
    print(f"Group Folds:        {args.n_splits} (Grouped by task_idx)")
    print("=" * 65)

    # 1. Load Dataset
    print("\n[1/3] Loading activation shards...")
    dataset = load_shards(args.data_dir)
    print(f"  -> Total turns: {dataset.num_samples}, Dimensions: {dataset.hidden_dim}")

    # 2. Round-wise AUC Trajectory (Figures 3 & 4 from the paper)
    print("\n[2/3] Evaluating Round-Wise Probes (GroupKFold on task_idx)...")
    prober = RoundwiseProber(dataset=dataset, C=args.C, n_splits=args.n_splits)
    round_results = prober.evaluate_auc_over_rounds(anchor=args.anchor, layer_idx=args.layer)

    print("\n" + "-" * 65)
    print(f"{'Round (t)':<12} {'ROC-AUC':<12} {'Samples':<12} {'Successes':<12} {'Failures':<12}")
    print("-" * 65)
    for res in round_results:
        print(f"Round {res.round_idx:<6d} {res.auc:<12.4f} {res.num_samples:<12d} {res.num_success:<12d} {res.num_fail:<12d}")
    print("-" * 65)

    # 3. Early-Abort Cascade Evaluation across Recall Targets
    print("\n[3/3] Evaluating Recall-Controlled Cascade...")
    cascade = RecallControlledCascade(
        confidence=args.confidence,
        n_splits=args.n_splits,
        C=args.C,
    )

    print("\n" + "=" * 65)
    print(f"{'Target Recall':<15} {'Actual Recall':<15} {'Aborted Failures':<18} {'Step Savings':<15}")
    print("=" * 65)
    for target in args.targets:
        res = cascade.evaluate_cascade(
            dataset=dataset,
            target_recall=target,
            anchor=args.anchor,
            layer_idx=args.layer,
        )
        print(
            f"{res.target_recall * 100:>5.1f}%          "
            f"{res.actual_recall * 100:>5.1f}%          "
            f"{res.aborted_failures_rate * 100:>5.1f}%             "
            f"{res.step_savings_rate * 100:>5.1f}%"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
