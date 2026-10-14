#!/usr/bin/env python
"""
Experiment 2: With a Standard Probe, Which Layer and Token Combination Performs Best?
  - Evaluates combinations of layers [4, 10, 16, 18, 20, 22, 24, 27] and anchors [post_gen, pre_gen, thought_end].
  - Computes performance:
      1. Within Benchmark (average in-domain 5-fold CV across all 4 benchmarks)
      2. Across Benchmark (average zero-shot transfer across all 12 directional transfer pairs)
  - Produces comparison heatmaps/charts saved to figures/exp2_layer_token_grid.png.
"""

from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import extract_episodes, load_shards
from src.standard_probe import StandardProbe


def main():
    benchmarks = ["bigcodebench", "textcraft", "intercode", "miniwob"]
    bench_names = {
        "bigcodebench": "BigCodeBench",
        "textcraft": "TextCraft",
        "intercode": "InterCode",
        "miniwob": "MiniWoB",
    }

    base_dir = Path(__file__).resolve().parent.parent / "data"
    data_paths = {
        "bigcodebench": base_dir / "bigcodebench_hf" / "features",
        "textcraft": base_dir / "textcraft_hf" / "features",
        "intercode": base_dir / "intercode_hf" / "features",
        "miniwob": base_dir / "miniwob_hf" / "features",
    }

    print("=" * 80)
    print("EXPERIMENT 2: OPTIMAL LAYER & TOKEN COMBINATION (WITHIN VS. ACROSS BENCHMARK)")
    print("=" * 80)

    # 1. Load Datasets
    print("[1/3] Loading activation shards for all benchmarks...")
    datasets = {b: load_shards(data_paths[b]) for b in benchmarks}
    layers = datasets["bigcodebench"].layer_ids
    anchors = ["post_gen", "pre_gen", "thought_end"]

    # Pre-extract episodes with all layers
    episodes_dict = {b: extract_episodes(datasets[b], target_layers=layers) for b in benchmarks}

    # 2. Grid Search over Layer x Anchor
    print(f"\n[2/3] Evaluating 24 configurations (8 layers x 3 anchors)...")
    results = []

    for anchor in anchors:
        print(f"  Anchor: {anchor}...")
        for l in layers:
            in_scores = []
            cross_scores = []

            # In-domain scores
            for b in benchmarks:
                probe = StandardProbe(layer=l, anchor=anchor)
                try:
                    _, cv_auc = probe.evaluate_cv(episodes_dict[b], n_splits=5)
                    in_scores.append(cv_auc * 100.0)
                except Exception:
                    continue

            # Cross-benchmark transfer scores
            for src in benchmarks:
                probe_src = StandardProbe(layer=l, anchor=anchor)
                try:
                    probe_src.fit(episodes_dict[src])
                except Exception:
                    continue

                for tgt in benchmarks:
                    if src == tgt:
                        continue
                    try:
                        m = probe_src.evaluate_classification(episodes_dict[tgt])
                        cross_scores.append(m.roc_auc * 100.0)
                    except Exception:
                        continue

            avg_in = float(np.mean(in_scores)) if in_scores else 0.0
            avg_cross = float(np.mean(cross_scores)) if cross_scores else 0.0

            results.append({
                "anchor": anchor,
                "layer": l,
                "within_benchmark_auc": avg_in,
                "across_benchmark_auc": avg_cross,
                "overall_avg_auc": (avg_in + avg_cross) / 2.0,
            })

    df = pd.DataFrame(results)

    # 3. Print Rankings
    print("\n" + "=" * 75)
    print("TOP LAYER & ANCHOR COMBINATIONS (WITHIN BENCHMARK)")
    print("=" * 75)
    top_within = df.sort_values(by="within_benchmark_auc", ascending=False).head(5)
    print(top_within[["anchor", "layer", "within_benchmark_auc", "across_benchmark_auc"]].round(2).to_string(index=False))

    print("\n" + "=" * 75)
    print("TOP LAYER & ANCHOR COMBINATIONS (ACROSS BENCHMARK / TRANSFER)")
    print("=" * 75)
    top_cross = df.sort_values(by="across_benchmark_auc", ascending=False).head(5)
    print(top_cross[["anchor", "layer", "across_benchmark_auc", "within_benchmark_auc"]].round(2).to_string(index=False))

    # 4. Generate Plot
    print("\n[3/3] Generating comparison charts...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    marker_map = {"post_gen": "o", "pre_gen": "s", "thought_end": "^"}
    color_map = {"post_gen": "#2980b9", "pre_gen": "#e67e22", "thought_end": "#27ae60"}

    # Left: Within Benchmark curves
    ax0 = axes[0]
    for anc in anchors:
        sub = df[df["anchor"] == anc].sort_values("layer")
        ax0.plot(
            sub["layer"],
            sub["within_benchmark_auc"],
            marker=marker_map[anc],
            linewidth=2.2,
            markersize=7,
            color=color_map[anc],
            label=f"Anchor: {anc}",
        )
    ax0.set_title("Within-Benchmark Performance (Average In-Domain 5-Fold CV)", fontsize=12, fontweight="bold", pad=10)
    ax0.set_xlabel("Transformer Layer", fontsize=11)
    ax0.set_ylabel("Average ROC-AUC (%)", fontsize=11)
    ax0.set_xticks(layers)
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(fontsize=10)

    # Right: Across Benchmark curves
    ax1 = axes[1]
    for anc in anchors:
        sub = df[df["anchor"] == anc].sort_values("layer")
        ax1.plot(
            sub["layer"],
            sub["across_benchmark_auc"],
            marker=marker_map[anc],
            linewidth=2.2,
            markersize=7,
            color=color_map[anc],
            label=f"Anchor: {anc}",
        )
    ax1.set_title("Across-Benchmark Transferability (Average Out-of-Domain Zero-Shot)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Transformer Layer", fontsize=11)
    ax1.set_ylabel("Average ROC-AUC (%)", fontsize=11)
    ax1.set_xticks(layers)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    plt.suptitle("Experiment 2: Layer & Token Anchor Performance (Within vs. Across Benchmark)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / "exp2_layer_token_grid.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✓ Saved plot to: {out_file}")

    df.to_csv(fig_dir / "exp2_layer_token_grid.csv", index=False)
    print(f"✓ Saved CSV to: {fig_dir / 'exp2_layer_token_grid.csv'}")


if __name__ == "__main__":
    main()
