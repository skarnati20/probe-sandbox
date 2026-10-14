#!/usr/bin/env python
"""
Experiment 4: How Does Within-Task (Mixed Rollout) vs. Global Matched Training Affect
In-Benchmark Generalization and Cross-Benchmark Transferability?
  - Isolates tasks that had both successes and failures across rollouts (Within-Task Mixed).
  - Matches sample size with random Global sampling having the exact same (N_succ, N_fail).
  - Evaluates performance on:
      1. Rest of the Benchmark (Held-out unseen unmixed tasks of same domain)
      2. Cross-Benchmark Transferability (Zero-shot transfer to other domains)
  - Produces multi-panel comparison charts saved to figures/exp4_within_vs_global.png.
"""

from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_shards


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
    print("EXPERIMENT 4: WITHIN-TASK MIXED VS. GLOBAL MATCHED PROBING")
    print("=" * 80)

    # 1. Load sliced data at layer 20, post_gen
    datasets = {b: load_shards(data_paths[b]) for b in benchmarks}
    data_slices = {b: datasets[b].get_slice(anchor="post_gen", layer_idx=20) for b in benchmarks}

    # Evaluate across sources with sufficient mixed tasks
    source_benches = ["bigcodebench", "textcraft", "intercode"]
    records = []

    for src in source_benches:
        X_src, y_src, groups_src = data_slices[src]
        unique_groups = np.unique(groups_src)
        mixed_groups = [g for g in unique_groups if len(np.unique(y_src[groups_src == g])) > 1]
        unmixed_groups = [g for g in unique_groups if g not in mixed_groups]

        mixed_mask = np.isin(groups_src, mixed_groups)
        unmixed_mask = np.isin(groups_src, unmixed_groups)

        X_mixed = X_src[mixed_mask]
        y_mixed = y_src[mixed_mask]

        n_pos = int(np.sum(y_mixed == 1))
        n_neg = int(np.sum(y_mixed == 0))

        # Train Within-Task probe
        pipe_within = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        pipe_within.fit(X_mixed, y_mixed)

        # Train 5 Global matched probes
        pos_idx = np.where(y_src == 1)[0]
        neg_idx = np.where(y_src == 0)[0]
        global_pipes = []
        for s in [42, 123, 456, 789, 1011]:
            rng = np.random.RandomState(s)
            sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            sel = np.concatenate([sel_pos, sel_neg])
            p_glob = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=s))
            p_glob.fit(X_src[sel], y_src[sel])
            global_pipes.append(p_glob)

        # 1. Evaluate on Rest of Same Benchmark (Held-out Unmixed Tasks)
        X_rest = X_src[unmixed_mask]
        y_rest = y_src[unmixed_mask]
        if len(np.unique(y_rest)) > 1:
            auc_within_rest = float(roc_auc_score(y_rest, pipe_within.predict_proba(X_rest)[:, 1]))
            auc_glob_rest = float(np.mean([roc_auc_score(y_rest, p.predict_proba(X_rest)[:, 1]) for p in global_pipes]))
            records.append({
                "source": bench_names[src],
                "eval_type": "Rest of Same Benchmark",
                "target": f"Rest of {bench_names[src]}",
                "within_auc": auc_within_rest * 100.0,
                "global_auc": auc_glob_rest * 100.0,
                "diff": (auc_within_rest - auc_glob_rest) * 100.0,
            })

        # 2. Evaluate on Other Benchmarks
        for tgt in benchmarks:
            if tgt == src:
                continue
            X_tgt, y_tgt, _ = data_slices[tgt]
            auc_within_tgt = float(roc_auc_score(y_tgt, pipe_within.predict_proba(X_tgt)[:, 1]))
            auc_glob_tgt = float(np.mean([roc_auc_score(y_tgt, p.predict_proba(X_tgt)[:, 1]) for p in global_pipes]))
            records.append({
                "source": bench_names[src],
                "eval_type": "Cross-Benchmark Transfer",
                "target": bench_names[tgt],
                "within_auc": auc_within_tgt * 100.0,
                "global_auc": auc_glob_tgt * 100.0,
                "diff": (auc_within_tgt - auc_glob_tgt) * 100.0,
            })

    df = pd.DataFrame(records)

    # Print Summary Table
    print("\n" + "=" * 90)
    print("WITHIN-TASK VS. GLOBAL MATCHED PROBE COMPARISON")
    print("=" * 90)
    print(df[["source", "eval_type", "target", "within_auc", "global_auc", "diff"]].round(2).to_string(index=False))

    # Generate Master Plot
    print("\nGenerating multi-panel comparison figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Rest of Benchmark In-Domain
    ax0 = axes[0]
    df_rest = df[df["eval_type"] == "Rest of Same Benchmark"]
    x0 = np.arange(len(df_rest))
    w = 0.35
    ax0.bar(x0 - w/2, df_rest["within_auc"], w, label="Within-Task Mixed", color="#e67e22", edgecolor="black")
    ax0.bar(x0 + w/2, df_rest["global_auc"], w, label="Global Matched", color="#2980b9", edgecolor="black")
    ax0.set_title("Rest of Same Benchmark (In-Domain)", fontsize=12, fontweight="bold", pad=10)
    ax0.set_ylabel("ROC-AUC (%)", fontsize=11)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(df_rest["source"], fontsize=10)
    ax0.set_ylim(50, 105)
    ax0.grid(axis="y", linestyle=":", alpha=0.6)
    ax0.legend(fontsize=9.5)

    # Subplot 2: Cross-Benchmark Transfer (Average across targets)
    ax1 = axes[1]
    df_cross = df[df["eval_type"] == "Cross-Benchmark Transfer"].groupby("source")[["within_auc", "global_auc"]].mean().reset_index()
    x1 = np.arange(len(df_cross))
    ax1.bar(x1 - w/2, df_cross["within_auc"], w, label="Within-Task Mixed", color="#e67e22", edgecolor="black")
    ax1.bar(x1 + w/2, df_cross["global_auc"], w, label="Global Matched", color="#2980b9", edgecolor="black")
    ax1.set_title("Average Cross-Benchmark Transfer (Zero-Shot)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Average ROC-AUC (%)", fontsize=11)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(df_cross["source"], fontsize=10)
    ax1.set_ylim(40, 95)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9.5)

    # Subplot 3: The Delta (Within-Task minus Global) across all individual transfer pairs
    ax2 = axes[2]
    pair_labels = [f"{r.source[:4]}->{r.target[:4]}" for _, r in df.iterrows()]
    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in df["diff"]]
    ax2.bar(pair_labels, df["diff"], color=colors, edgecolor="black")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("Performance Difference (Within-Task - Global)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("ROC-AUC Delta (%)", fontsize=11)
    ax2.set_xticks(range(len(pair_labels)))
    ax2.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Experiment 4: Within-Task (Mixed) vs. Global Matched Training Performance", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / "exp4_within_vs_global.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✓ Saved plot to: {out_file}")

    df.to_csv(fig_dir / "exp4_within_vs_global.csv", index=False)
    print(f"✓ Saved CSV to: {fig_dir / 'exp4_within_vs_global.csv'}")


if __name__ == "__main__":
    main()
