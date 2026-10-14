#!/usr/bin/env python
"""
Experiment 3: Cross-Benchmark Transfer for All 6 Canonical Probes
  - Trains each of the 6 probe architectures on EVERY benchmark (BigCodeBench, TextCraft, InterCode, MiniWoB).
  - Evaluates on all 4 benchmarks (In-domain 5-fold CV vs. Across-benchmark zero-shot transfer).
  - Produces a 4-chart transfer plot for EACH probe method (matching the layout of Exp 1):
      1. exp3_standard_probe_transfer_4charts.png
      2. exp3_doomed_probe_transfer_4charts.png
      3. exp3_step_delta_probe_transfer_4charts.png
      4. exp3_token_delta_probe_transfer_4charts.png
      5. exp3_cosine_drift_probe_transfer_4charts.png
      6. exp3_many_layer_probe_transfer_4charts.png
  - Saves a comprehensive cross-benchmark comparison CSV: exp3_probe_comparison_all_benches.csv.
"""

from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cosine_drift_probe import CosineDriftProbe
from src.data import extract_episodes, load_shards
from src.doomed_probe import DoomedProbe
from src.many_layer_probe import ManyLayerProbe
from src.standard_probe import StandardProbe
from src.step_delta_probe import StepDeltaProbe
from src.token_delta_probe import TokenDeltaProbe


def create_probe(probe_id: str, layer: int, layers_all: list[int]):
    if probe_id == "standard":
        return StandardProbe(layer=layer, anchor="post_gen")
    elif probe_id == "doomed":
        return DoomedProbe(layer=layer, anchor="post_gen")
    elif probe_id == "step_delta":
        return StepDeltaProbe(layer=layer, anchor="post_gen")
    elif probe_id == "token_delta":
        return TokenDeltaProbe(layer=layer)
    elif probe_id == "cosine_drift":
        return CosineDriftProbe(layer=layer, anchor="post_gen")
    elif probe_id == "many_layer":
        return ManyLayerProbe(layers=layers_all, anchor="post_gen")
    else:
        raise ValueError(f"Unknown probe: {probe_id}")


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

    probe_configs = [
        ("standard", "Standard Probe", "exp3_standard_probe_transfer_4charts.png"),
        ("doomed", "Doomed Probe (Paper Cascade)", "exp3_doomed_probe_transfer_4charts.png"),
        ("step_delta", "Step Delta Probe", "exp3_step_delta_probe_transfer_4charts.png"),
        ("token_delta", "Token Delta Probe (Post-Pre)", "exp3_token_delta_probe_transfer_4charts.png"),
        ("cosine_drift", "Cosine Drift Probe", "exp3_cosine_drift_probe_transfer_4charts.png"),
        ("many_layer", "Many Layer Input Probe", "exp3_many_layer_probe_transfer_4charts.png"),
    ]

    print("=" * 80)
    print("EXPERIMENT 3: CROSS-BENCHMARK TRANSFER ACROSS ALL 6 PROBE METHODS")
    print("=" * 80)

    # 1. Load all datasets & episodes
    print("\n[1/3] Loading activation shards for all benchmarks...")
    datasets = {b: load_shards(data_paths[b]) for b in benchmarks}
    layers_all = datasets["bigcodebench"].layer_ids
    episodes_dict = {b: extract_episodes(datasets[b], target_layers=layers_all) for b in benchmarks}

    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    # 2. Loop over each probe method
    print("\n[2/3] Evaluating 4x4 transfer matrices for each probe method...")
    for probe_id, probe_name, plot_filename in probe_configs:
        print(f"\n--- Running: {probe_name} ---")
        matrix = pd.DataFrame(index=benchmarks, columns=benchmarks, dtype=float)

        for src in benchmarks:
            src_eps = episodes_dict[src]
            train_probe = create_probe(probe_id, layer=20, layers_all=layers_all)

            # Fit on full source
            try:
                if isinstance(train_probe, DoomedProbe):
                    train_probe.fit(src_eps, target_recall=0.95)
                else:
                    train_probe.fit(src_eps)
            except Exception as e:
                print(f"    [Warning fitting on {src}: {e}]")

            for tgt in benchmarks:
                tgt_eps = episodes_dict[tgt]
                if src == tgt:
                    # In-domain 5-fold CV
                    eval_probe = create_probe(probe_id, layer=20, layers_all=layers_all)
                    try:
                        if isinstance(eval_probe, DoomedProbe):
                            m, _ = eval_probe.evaluate_cv(tgt_eps, n_splits=5, target_recall=0.95)
                            auc = m.roc_auc
                        else:
                            _, auc = eval_probe.evaluate_cv(tgt_eps, n_splits=5)
                    except Exception as e:
                        auc = 0.5
                else:
                    # Zero-shot cross-benchmark transfer
                    try:
                        m = train_probe.evaluate_classification(tgt_eps)
                        auc = m.roc_auc
                    except Exception as e:
                        auc = 0.5

                matrix.loc[src, tgt] = auc * 100.0

        # Print matrix for this probe
        print(f"Transfer ROC-AUC Matrix (%) for {probe_name}:")
        print(matrix.round(2))

        # Record summary metrics
        in_scores = [matrix.loc[b, b] for b in benchmarks]
        cross_scores = [matrix.loc[src, tgt] for src in benchmarks for tgt in benchmarks if src != tgt]
        summary_records.append({
            "probe_id": probe_id,
            "probe_name": probe_name,
            "avg_in_benchmark_auc": float(np.mean(in_scores)),
            "avg_across_benchmark_auc": float(np.mean(cross_scores)),
            "bigcode_transfer_avg": float(np.mean([matrix.loc["bigcodebench", tgt] for tgt in benchmarks if tgt != "bigcodebench"])),
            "textcraft_transfer_avg": float(np.mean([matrix.loc["textcraft", tgt] for tgt in benchmarks if tgt != "textcraft"])),
            "intercode_transfer_avg": float(np.mean([matrix.loc["intercode", tgt] for tgt in benchmarks if tgt != "intercode"])),
            "miniwob_transfer_avg": float(np.mean([matrix.loc["miniwob", tgt] for tgt in benchmarks if tgt != "miniwob"])),
        })

        # Generate 4-chart plot for this probe (matching Exp 1 layout)
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
            ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.grid(axis="y", linestyle=":", alpha=0.6)

            if i == 0:
                legend_elements = [
                    Patch(facecolor=color_in, edgecolor="black", label="In-Benchmark (5-Fold CV)"),
                    Patch(facecolor=colors_cross, edgecolor="black", label="Across-Benchmark (Zero-Shot)"),
                ]
                ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

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
            f"Experiment 3: {probe_name} Transfer (In-Benchmark vs. Across-Benchmark)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = fig_dir / plot_filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  ✓ Saved 4-chart plot to: {out_path}")

    # 3. Save Overall Summary CSV
    df_summary = pd.DataFrame(summary_records)
    csv_path = fig_dir / "exp3_probe_comparison_all_benches.csv"
    df_summary.to_csv(csv_path, index=False)
    print("\n" + "=" * 90)
    print("ALL 6 PROBES TRANSFER SUMMARY ACROSS ALL TRAINING BENCHMARKS")
    print("=" * 90)
    print(df_summary[["probe_name", "avg_in_benchmark_auc", "avg_across_benchmark_auc", "bigcode_transfer_avg", "textcraft_transfer_avg"]].round(2).to_string(index=False))
    print(f"\n✓ Saved overall summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()
