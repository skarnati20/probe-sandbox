#!/usr/bin/env python
"""
Benchmarks the 6 canonical probe architectures under 5-Fold Cross-Validation:
  1. Standard Probe
  2. Doomed Probe (Paper Cascade)
  3. Step Delta Probe (Vector difference with previous hidden state)
  4. Token Delta Probe (Post-Gen - Pre-Gen token difference)
  5. Cosine Drift Probe (Angular and directional trajectory kinematics)
  6. Many Layer Input Probe (Multi-layer residual stream fusion)
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.cosine_drift_probe import CosineDriftProbe
from src.data import extract_episodes, load_shards
from src.doomed_probe import DoomedProbe
from src.many_layer_probe import ManyLayerProbe
from src.standard_probe import StandardProbe
from src.step_delta_probe import StepDeltaProbe
from src.token_delta_probe import TokenDeltaProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 6 Canonical Probes")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/textcraft_hf",
        help="Path to dataset directory containing features/ (default: data/textcraft_hf)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Primary target layer for single-layer probes (default: 20)",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.95,
        help="Target success preservation recall (default: 0.95)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save comparison results CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir)
    feat_path = data_path / "features" if (data_path / "features").exists() else data_path

    print("=" * 80)
    print("EMPIRICAL BENCHMARK: 6 CANONICAL PROBES EVALUATION")
    print(f"Data Directory:         {data_path}")
    print(f"Primary Layer:          {args.layer}")
    print(f"Target Success Recall:  >={args.target_recall * 100:.1f}%")
    print(f"CV Folds:               {args.n_splits}")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[1/2] Loading activation shards & extracting episodes...")
    dataset = load_shards(feat_path)
    episodes = extract_episodes(dataset)
    print(f"  -> Loaded {len(episodes)} episodes across {dataset.hidden_dim} dimensions.")

    # 2. Define the 6 Probes
    probes = [
        StandardProbe(layer=args.layer, anchor="post_gen"),
        DoomedProbe(layer=args.layer, anchor="post_gen"),
        StepDeltaProbe(layer=args.layer, anchor="post_gen"),
        TokenDeltaProbe(layer=args.layer),
        CosineDriftProbe(layer=args.layer, anchor="post_gen"),
        ManyLayerProbe(layers=dataset.layer_ids, anchor="post_gen"),
    ]

    # 3. Evaluate each probe
    print("\n[2/2] Running 5-Fold Cross-Validation across all 6 probes...")
    records = []

    for probe in probes:
        print(f"  -> Evaluating {probe.name}...")
        try:
            if isinstance(probe, DoomedProbe):
                metrics, casc = probe.evaluate_cv(
                    episodes, n_splits=args.n_splits, target_recall=args.target_recall
                )
            else:
                metrics, _ = probe.evaluate_cv(episodes, n_splits=args.n_splits)
                # Calibrate on full and evaluate cascade
                probe.fit(episodes)
                casc = probe.evaluate_cascade(episodes, target_recall=args.target_recall)

            records.append({
                "probe": probe.name,
                "roc_auc": metrics.roc_auc,
                "accuracy": metrics.accuracy,
                "recall_preserved": casc.recall_preserved,
                "false_abort_rate": casc.false_abort_rate,
                "doomed_abort_rate": casc.doomed_abort_rate,
                "step_savings": casc.step_savings,
                "avg_abort_step": casc.avg_abort_step,
            })
        except Exception as e:
            print(f"     [Error evaluating {probe.name}: {e}]")

    # 4. Display Results Table
    df = pd.DataFrame(records)
    print("\n" + "=" * 95)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 95)
    fmt_df = pd.DataFrame()
    fmt_df["Probe"] = df["probe"]
    fmt_df["ROC-AUC"] = (df["roc_auc"] * 100).map("{:.2f}%".format)
    fmt_df["Accuracy"] = (df["accuracy"] * 100).map("{:.2f}%".format)
    fmt_df["Success Recall"] = (df["recall_preserved"] * 100).map("{:.1f}%".format)
    fmt_df["Doomed Abort"] = (df["doomed_abort_rate"] * 100).map("{:.1f}%".format)
    fmt_df["Step Savings"] = (df["step_savings"] * 100).map("{:.1f}%".format)
    fmt_df["Avg Abort Step"] = df["avg_abort_step"].map("{:.2f}".format)
    print(fmt_df.to_string(index=False))
    print("=" * 95)

    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"Saved benchmark results to {out_p}")


if __name__ == "__main__":
    main()
