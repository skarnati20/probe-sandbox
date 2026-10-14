#!/usr/bin/env python
"""
Experiment 1: How Do Standard Probes Transfer In-Benchmark vs. Across Benchmark
  - Trains on each of the 4 benchmarks (BigCodeBench, TextCraft, InterCode, MiniWoB).
  - Evaluates on all 4 benchmarks (In-domain 5-fold CV vs. Across-benchmark zero-shot).
  - Generates a single figure containing 4 separate charts (one for each training benchmark).
"""

from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import extract_episodes, load_shards
from src.standard_probe import StandardProbe


def main():
    benchmarks = ["bigcodebench", "textcraft", "intercode", "miniwob"]
    bench_names = {
        "bigcodebench": "BigCodeBench",
        "textcraft": "TextCraft",
        "intercode": "InterCode-Bash",
        "miniwob": "MiniWoB++",
    }

    base_dir = Path(__file__).resolve().parent.parent / "data"
    data_paths = {
        "bigcodebench": base_dir / "bigcodebench_hf" / "features",
        "textcraft": base_dir / "textcraft_hf" / "features",
        "intercode": base_dir / "intercode_hf" / "features",
        "miniwob": base_dir / "miniwob_hf" / "features",
    }

    print("=" * 80)
    print("EXPERIMENT 1: STANDARD PROBE TRANSFER (IN-BENCHMARK VS. ACROSS-BENCHMARK)")
    print("=" * 80)

    # 1. Load all datasets and episodes
    print("[1/3] Loading activation shards for all 4 benchmarks...")
    episodes_dict = {}
    for b in benchmarks:
        ds = load_shards(data_paths[b])
        eps = extract_episodes(ds, target_layers=[20])
        episodes_dict[b] = eps
        print(f"  -> {bench_names[b]}: {len(eps)} episodes loaded.")

    # 2. Compute 4x4 matrix
    print("\n[2/3] Computing transfer matrix across all benchmark pairs...")
    matrix = pd.DataFrame(index=benchmarks, columns=benchmarks, dtype=float)

    for src in benchmarks:
        src_eps = episodes_dict[src]
        probe = StandardProbe(layer=20, anchor="post_gen")
        probe.fit(src_eps)

        for tgt in benchmarks:
            tgt_eps = episodes_dict[tgt]
            if src == tgt:
                # 5-fold CV for in-domain
                _, auc = probe.evaluate_cv(tgt_eps, n_splits=5)
            else:
                metrics = probe.evaluate_classification(tgt_eps)
                auc = metrics.roc_auc

            matrix.loc[src, tgt] = auc * 100.0

    print("\n" + "=" * 65)
    print("TRANSFER ROC-AUC (%) MATRIX")
    print("=" * 65)
    print(matrix.round(2))
    print("=" * 65)

    # 3. Plot 4 charts in one master figure
    print("\n[3/3] Generating 4-chart transfer figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors_cross = "#3498db"
    color_in = "#2ecc71"

    for i, src in enumerate(benchmarks):
        ax = axes[i]
        src_label = bench_names[src]
        scores = [matrix.loc[src, tgt] for tgt in benchmarks]
        labels = [bench_names[tgt] for tgt in benchmarks]
        bar_colors = [color_in if src == tgt else colors_cross for tgt in benchmarks]

        bars = ax.bar(labels, scores, color=bar_colors, width=0.55, edgecolor="black", linewidth=1)
        ax.set_title(f"Trained on: {src_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("ROC-AUC (%)", fontsize=11)
        ax.set_ylim(0, 110)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance Level (50%)")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Highlight in-benchmark vs across-benchmark in legend for first chart
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color_in, edgecolor="black", label="In-Benchmark (5-Fold CV)"),
                Patch(facecolor=colors_cross, edgecolor="black", label="Across-Benchmark (Zero-Shot)"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        # Annotate bar values
        for bar, val in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 2.0,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.suptitle(
        "Experiment 1: Standard Probe Transfer (In-Benchmark vs. Across-Benchmark)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / "exp1_standard_transfer_4charts.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✓ Saved 4-chart transfer figure to: {out_file}")

    # Also save CSV summary
    matrix.to_csv(fig_dir / "exp1_transfer_matrix.csv")
    print(f"✓ Saved transfer matrix CSV to: {fig_dir / 'exp1_transfer_matrix.csv'}")


if __name__ == "__main__":
    main()
