#!/usr/bin/env python
"""
Unified CLI to train and evaluate any of the 6 canonical probes:
  1. standard     : Standard Probe (Static h_t^post)
  2. doomed       : Doomed Probe (Paper Cascade Probe with Clopper-Pearson recall gates)
  3. step_delta   : Step Delta Probe (h_t^post - h_{t-1}^post)
  4. token_delta  : Token Delta Probe (h_t^post - h_t^pre)
  5. cosine_drift : Cosine Drift Probe (Directional and angular kinematics)
  6. many_layer   : Many Layer Input Probe (Multi-layer residual stream fusion)
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cosine_drift_probe import CosineDriftProbe
from src.data import extract_episodes, load_shards
from src.doomed_probe import DoomedProbe
from src.many_layer_probe import ManyLayerProbe
from src.standard_probe import StandardProbe
from src.step_delta_probe import StepDeltaProbe
from src.token_delta_probe import TokenDeltaProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate a Canonical Probe")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing features/",
    )
    parser.add_argument(
        "--probe",
        type=str,
        default="standard",
        choices=["standard", "doomed", "step_delta", "token_delta", "cosine_drift", "many_layer"],
        help="Probe architecture to train (default: standard)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Primary target layer for single-layer probes (default: 20)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer list for ManyLayerProbe (default: all available)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="post_gen",
        help="Token anchor for probes that support it (default: post_gen)",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.95,
        help="Target recall preservation guarantee (default: 0.95)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir)
    feat_path = data_path / "features" if (data_path / "features").exists() else data_path

    print("=" * 65)
    print("CANONICAL PROBE TRAINING & EVALUATION")
    print(f"Data Directory:         {data_path}")
    print(f"Probe:                  {args.probe.upper()}")
    print(f"Layer:                  {args.layer}")
    print(f"Target Recall:          {args.target_recall * 100:.1f}%")
    print(f"Regularization C:       {args.C}")
    print("=" * 65)

    # 1. Load dataset & extract episodes
    print("\n[1/2] Loading activation shards & extracting episodes...")
    dataset = load_shards(feat_path)
    episodes = extract_episodes(dataset)
    print(f"  -> Loaded {len(episodes)} episodes across {dataset.hidden_dim} dimensions.")

    # 2. Instantiate selected probe
    if args.probe == "standard":
        probe = StandardProbe(layer=args.layer, anchor=args.anchor, C=args.C)
    elif args.probe == "doomed":
        probe = DoomedProbe(layer=args.layer, anchor=args.anchor, C=args.C)
    elif args.probe == "step_delta":
        probe = StepDeltaProbe(layer=args.layer, anchor=args.anchor, C=args.C)
    elif args.probe == "token_delta":
        probe = TokenDeltaProbe(layer=args.layer, C=args.C)
    elif args.probe == "cosine_drift":
        probe = CosineDriftProbe(layer=args.layer, anchor=args.anchor, C=args.C)
    elif args.probe == "many_layer":
        layers_to_use = args.layers or dataset.layer_ids
        probe = ManyLayerProbe(layers=layers_to_use, anchor=args.anchor, C=args.C)
    else:
        raise ValueError(f"Unknown probe: {args.probe}")

    # 3. Evaluate 5-fold CV & Cascade
    print(f"\n[2/2] Running 5-Fold Cross-Validation for {probe.name}...")
    if isinstance(probe, DoomedProbe):
        metrics, casc = probe.evaluate_cv(
            episodes, n_splits=args.n_splits, target_recall=args.target_recall
        )
    else:
        metrics, _ = probe.evaluate_cv(episodes, n_splits=args.n_splits)
        probe.fit(episodes)
        casc = probe.evaluate_cascade(episodes, target_recall=args.target_recall)

    print("\n" + "=" * 65)
    print(f"RESULTS FOR {probe.name.upper()}")
    print("=" * 65)
    print(f"Classification ROC-AUC:         {metrics.roc_auc * 100:.2f}%")
    print(f"Accuracy:                       {metrics.accuracy * 100:.2f}%")
    print(f"Balanced Accuracy:              {metrics.balanced_accuracy * 100:.2f}%")
    print(f"Precision:                      {metrics.precision:.4f}")
    print(f"Recall:                         {metrics.recall:.4f}")
    print(f"F1 Score:                       {metrics.f1:.4f}")
    print("-" * 65)
    print(f"Success Recall Preserved:       {casc.recall_preserved * 100:.1f}%")
    print(f"False Abort Rate:               {casc.false_abort_rate * 100:.1f}%")
    print(f"Doomed Episodes Aborted:        {casc.doomed_abort_rate * 100:.1f}%")
    print(f"Step Savings Rate:              {casc.step_savings * 100:.1f}%")
    print(f"Average Abort Step:             {casc.avg_abort_step:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
