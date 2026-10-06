from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from .data import ShardDataset
from .roundwise_probe import fit_oof_logistic


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """
    Exact Clopper-Pearson binomial lower confidence bound.
    Given k successes out of n trials, returns the lower bound for true proportion p.
    """
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    alpha = 1.0 - confidence
    return float(stats.beta.ppf(alpha, k, n - k + 1))


@dataclass
class CascadeGate:
    round_idx: int
    threshold: float
    target_recall: float
    calibrated_lower_bound: float
    num_calibration_samples: int


@dataclass
class CascadeEvaluationResult:
    target_recall: float
    actual_recall: float            # Proportion of successful episodes preserved
    aborted_failures_rate: float    # Proportion of doomed episodes aborted early
    step_savings_rate: float        # Percentage of total steps saved by aborting early
    mean_steps_aborted: float       # Average steps taken by aborted failure episodes
    mean_steps_unaborted: float     # Average steps taken without early abort
    total_episodes: int
    total_successes: int
    total_failures: int
    gates: list[CascadeGate]


class RecallControlledCascade:
    """
    Early Abort Cascade with exact Clopper-Pearson calibration.
    Faithfully reproduces the method and certification in 'Doomed from the Start' (2026).
    """

    def __init__(
        self,
        confidence: float = 0.95,
        n_splits: int = 5,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.confidence = confidence
        self.n_splits = n_splits
        self.C = C
        self.random_state = random_state

    def calibrate_gate_threshold(
        self,
        scores_pos: np.ndarray,
        target_recall: float,
    ) -> tuple[float, float]:
        """
        Finds the maximum threshold tau such that the Clopper-Pearson lower bound >= target_recall.
        """
        if len(scores_pos) == 0:
            return -np.inf, 0.0

        sorted_scores = np.sort(scores_pos)
        n = len(sorted_scores)

        best_tau = -np.inf
        best_lb = 0.0

        for tau in sorted_scores:
            k = int(np.sum(sorted_scores >= tau))
            lb = clopper_pearson_lower(k=k, n=n, confidence=self.confidence)
            if lb >= target_recall:
                best_tau = tau
                best_lb = lb
            else:
                break

        # Fallback to minimum score if conservative bound is tight
        if best_tau == -np.inf:
            best_tau = sorted_scores[0]
            best_lb = clopper_pearson_lower(n, n, confidence=self.confidence)

        return float(best_tau), float(best_lb)

    def evaluate_cascade(
        self,
        dataset: ShardDataset,
        target_recall: float = 0.95,
        anchor: str = "post_gen",
        layer_idx: Optional[int] = None,
        max_round: Optional[int] = None,
    ) -> CascadeEvaluationResult:
        """
        Simulates and evaluates the full recall-controlled cascade across all rounds.
        """
        avail_rounds = sorted(np.unique(dataset.rounds))
        if max_round is not None:
            avail_rounds = [r for r in avail_rounds if r <= max_round]

        # 1. Gather all unique episodes and their ground-truth length & success
        ep_dict: dict[str, dict] = {}
        for m in dataset.metas:
            ep_id = m.get("episode_id", f"{m['task_idx']}_{m['rollout_k']}")
            if ep_id not in ep_dict:
                ep_dict[ep_id] = {
                    "episode_id": ep_id,
                    "task_idx": m["task_idx"],
                    "rollout_k": m["rollout_k"],
                    "success": m["success"],
                    "n_rounds": m.get("n_rounds", 1),
                    "round_scores": {},
                }

        # 2. Train per-round OOF probes and calibrate gate thresholds
        gates: list[CascadeGate] = []
        for t in avail_rounds:
            try:
                X_t, y_t, groups_t = dataset.get_slice(
                    anchor=anchor,
                    round_idx=t,
                    layer_idx=layer_idx,
                )
            except ValueError:
                continue

            # Need at least 2 positive samples to calibrate gates
            if len(np.unique(y_t)) < 2 or np.sum(y_t == 1) < 2:
                continue

            scores_t = fit_oof_logistic(
                X_t,
                y_t,
                groups=groups_t,
                n_splits=self.n_splits,
                C=self.C,
                random_state=self.random_state,
            )

            # Record round score per episode
            idx_in_slice = 0
            for m in dataset.metas:
                if m.get("anchor", "post_gen") == anchor and m.get("round", m.get("step_idx", 0)) == t:
                    ep_id = m.get("episode_id", f"{m['task_idx']}_{m['rollout_k']}")
                    if ep_id in ep_dict:
                        ep_dict[ep_id]["round_scores"][t] = scores_t[idx_in_slice]
                    idx_in_slice += 1

            # Calibrate threshold on positive (successful) validation scores
            pos_scores = scores_t[y_t == 1]
            tau_t, lb_t = self.calibrate_gate_threshold(pos_scores, target_recall=target_recall)
            gates.append(
                CascadeGate(
                    round_idx=t,
                    threshold=tau_t,
                    target_recall=target_recall,
                    calibrated_lower_bound=lb_t,
                    num_calibration_samples=len(pos_scores),
                )
            )

        # 3. Simulate episode-level cascade execution
        gate_lookup = {g.round_idx: g.threshold for g in gates}
        total_episodes = len(ep_dict)
        n_successes = sum(1 for ep in ep_dict.values() if ep["success"] == 1)
        n_failures = total_episodes - n_successes

        survived_successes = 0
        aborted_failures = 0

        baseline_total_steps = sum(ep["n_rounds"] for ep in ep_dict.values())
        cascade_total_steps = 0

        for ep in ep_dict.values():
            aborted_at = None
            for t in sorted(ep["round_scores"].keys()):
                score = ep["round_scores"][t]
                if t in gate_lookup and score < gate_lookup[t]:
                    aborted_at = t + 1  # Aborted at turn t (executed t+1 steps)
                    break

            if aborted_at is not None:
                cascade_total_steps += aborted_at
                if ep["success"] == 0:
                    aborted_failures += 1
            else:
                cascade_total_steps += ep["n_rounds"]
                if ep["success"] == 1:
                    survived_successes += 1

        actual_recall = (survived_successes / max(1, n_successes)) if n_successes > 0 else 1.0
        aborted_failures_rate = (aborted_failures / max(1, n_failures)) if n_failures > 0 else 0.0
        step_savings = max(0.0, (baseline_total_steps - cascade_total_steps) / max(1, baseline_total_steps))

        return CascadeEvaluationResult(
            target_recall=target_recall,
            actual_recall=actual_recall,
            aborted_failures_rate=aborted_failures_rate,
            step_savings_rate=step_savings,
            mean_steps_aborted=cascade_total_steps / max(1, total_episodes),
            mean_steps_unaborted=baseline_total_steps / max(1, total_episodes),
            total_episodes=total_episodes,
            total_successes=n_successes,
            total_failures=n_failures,
            gates=gates,
        )
