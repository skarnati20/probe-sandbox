#!/usr/bin/env python
"""
Master Experiment Runner:
Executes all 4 canonical probe experiments and generates all figures:
  1. Experiment 1: Standard Probe Transfer (In-Benchmark vs. Across-Benchmark, 4-chart plot)
  2. Experiment 2: Layer & Token Anchor Performance (Within-Benchmark vs. Across-Benchmark)
  3. Experiment 3: 6 Canonical Probes Performance (In-Benchmark vs. Across-Benchmark)
  4. Experiment 4: Within-Task Mixed vs. Global Matched Training Performance
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path):
    print("\n" + "#" * 80)
    print(f"# RUNNING: {script_path.name}")
    print("#" * 80)
    cmd = [sys.executable, str(script_path)]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error running {script_path.name}")
        sys.exit(res.returncode)


def main():
    scripts_dir = Path(__file__).resolve().parent

    experiments = [
        scripts_dir / "exp1_transfer_matrix.py",
        scripts_dir / "exp2_layer_token_grid.py",
        scripts_dir / "exp3_probe_comparison.py",
        scripts_dir / "exp4_within_vs_global.py",
    ]

    print("=" * 80)
    print("RUNNING ALL 4 EXPERIMENTAL EVALUATION SUITES")
    print("=" * 80)

    for exp_script in experiments:
        run_script(exp_script)

    print("\n" + "=" * 80)
    print("✓ ALL 4 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"✓ All plots and CSV summaries generated in: {scripts_dir.parent / 'figures'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
