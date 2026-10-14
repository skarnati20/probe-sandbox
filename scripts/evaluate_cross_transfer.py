#!/usr/bin/env python
"""
Cross-Benchmark Zero-Shot Transfer Evaluation for the 6 Canonical Probes:
Trains each probe architecture on a Source benchmark (e.g. BigCodeBench or TextCraft)
and evaluates zero-shot out-of-domain transfer on a Target benchmark (e.g. MiniWoB++ or InterCode-Bash).

Evaluates:
  1. Standard Probe
  2. Doomed Probe (Paper Cascade)
  3. Step Delta Probe (Previous hidden state difference)
  4. Token Delta Probe (Post-Gen - Pre-Gen difference)
  5. Cosine Drift Probe (Directional and angular kinematics)
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
    parser = argparse.ArgumentParser(description="Cross-Benchmark Transfer for 6 Probes")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/bigcodebench_hf",
        help="Path to source dataset directory (Train)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/miniwob_hf",
        help="Path to target dataset directory (Zero-Shot Test)",
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
    return parser.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.source_dir)
    tgt_path = Path(args.target_dir)

    src_feat = src_path / "features" if (src_path / "features").exists() else src_path
    tgt_feat = tgt_path / "features" if (tgt_path / "features").exists() else tgt_path

    print("=" * 80)
    print("CROSS-BENCHMARK ZERO-SHOT TRANSFER EVALUATION")
    print(f"Source Benchmark (Train): {src_path.name}")
    print(f"Target Benchmark (Test):  {tgt_path.name}")
    print(f"Layer:                    {args.layer}")
    print(f"Target Success Recall:    >={args.target_recall * 100:.1f}%")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1/3] Loading source & target shards...")
    src_ds = load_shards(src_feat)
    tgt_ds = load_shards(tgt_feat)

    src_eps = extract_episodes(src_ds)
    tgt_eps = extract_episodes(tgt_ds)
    print(f"  -> Source: {len(src_eps)} episodes (Successes: {sum(e['success'] for e in src_eps)})")
    print(f"  -> Target: {len(tgt_eps)} episodes (Successes: {sum(e['success'] for e in tgt_eps)})")

    # Shared layers for ManyLayerProbe
    shared_layers = sorted(list(set(src_ds.layer_ids).intersection(set(tgt_ds.layer_ids))))

    # 2. Define the 6 Probes
    probes = [
        StandardProbe(layer=args.layer, anchor="post_gen"),
        DoomedProbe(layer=args.layer, anchor="post_gen"),
        StepDeltaProbe(layer=args.layer, anchor="post_gen"),
        TokenDeltaProbe(layer=args.layer),
        CosineDriftProbe(layer=args.layer, anchor="post_gen"),
        ManyLayerProbe(layers=shared_layers, anchor="post_gen"),
    ]

    # 3. Train on Source & Evaluate Zero-Shot on Target
    print("\n[2/3] Training on Source & Evaluating Zero-Shot on Target...")
    records = []

    for probe in probes:
        print(f"  -> Transferring {probe.name}...")
        try:
            if isinstance(probe, DoomedProbe):
                probe.fit(src_eps, target_recall=args.target_recall)
            else:
                probe.fit(src_eps)

            # Evaluate zero-shot classification on target
            metrics = probe.evaluate_classification(tgt_eps)
            # Evaluate zero-shot cascade on target
            casc = probe.evaluate_cascade(tgt_eps, target_recall=args.target_recall)

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
            print(f"     [Error during transfer of {probe.name}: {e}]")

    # 4. Display Results
    df = pd.DataFrame(records)
    print("\n" + "=" * 95)
    print(f"ZERO-SHOT TRANSFER RESULTS: {src_path.name.upper()} ──► {tgt_path.name.upper()}")
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


if __name__ == "__main__":
    main()
