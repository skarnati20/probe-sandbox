"""
Base definitions, metrics, and calibration functions for LLM agent probes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """
    Computes exact Clopper-Pearson binomial lower confidence bound.
    Given k preserved successes out of n total successes, returns the lower bound
    for the true success preservation rate at the given confidence level.
    """
    if n == 0 or k == 0:
        return 0.0
    alpha = 1.0 - confidence
    return float(stats.beta.ppf(alpha, k, n - k + 1))


@dataclass
class ProbeMetrics:
    roc_auc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int
    num_positives: int
    num_negatives: int


@dataclass
class CascadeMetrics:
    recall_preserved: float       # Proportion of successful episodes preserved
    false_abort_rate: float       # 1.0 - recall_preserved
    doomed_abort_rate: float      # Proportion of doomed (failed) episodes aborted early
    step_savings: float           # Percentage of total execution steps saved
    avg_abort_step: float         # Average round where doomed episodes were aborted
    total_episodes: int
    total_successes: int
    total_failures: int
    target_recall: float


def evaluate_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ProbeMetrics:
    """Computes comprehensive binary classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = 0.5

    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return ProbeMetrics(
        roc_auc=auc,
        accuracy=acc,
        balanced_accuracy=bacc,
        precision=prec,
        recall=rec,
        f1=f1,
        num_samples=len(y_true),
        num_positives=int(np.sum(y_true == 1)),
        num_negatives=int(np.sum(y_true == 0)),
    )


def evaluate_abort_policy(
    episodes: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any], int], Optional[float]],
    threshold_fn: Callable[[int], Optional[float]],
    target_recall: float = 0.95,
) -> CascadeMetrics:
    """
    Simulates early abort cascade execution across multi-turn agent episodes.
    At each step r, if score_fn(ep, r) < threshold_fn(r), the episode is aborted.
    """
    total_succ = sum(1 for ep in episodes if ep["success"] == 1)
    total_fail = sum(1 for ep in episodes if ep["success"] == 0)

    succ_preserved = 0
    fail_aborted = 0

    total_baseline_steps = sum(len(ep["rounds"]) for ep in episodes)
    total_executed_steps = 0
    abort_steps_list = []

    for ep in episodes:
        rounds = sorted(ep["rounds"].keys())
        aborted = False
        abort_round = len(rounds)

        for step in rounds:
            score = score_fn(ep, step)
            thresh = threshold_fn(step)
            if score is not None and thresh is not None and score < thresh:
                aborted = True
                abort_round = step + 1
                break

        total_executed_steps += abort_round

        if ep["success"] == 1:
            if not aborted:
                succ_preserved += 1
        else:
            if aborted:
                fail_aborted += 1
                abort_steps_list.append(abort_round)

    actual_recall = succ_preserved / max(1, total_succ)
    doomed_rate = fail_aborted / max(1, total_fail)
    step_savings = 1.0 - (total_executed_steps / max(1, total_baseline_steps))
    avg_abort_step = float(np.mean(abort_steps_list)) if abort_steps_list else 0.0

    return CascadeMetrics(
        recall_preserved=actual_recall,
        false_abort_rate=1.0 - actual_recall,
        doomed_abort_rate=doomed_rate,
        step_savings=step_savings,
        avg_abort_step=avg_abort_step,
        total_episodes=len(episodes),
        total_successes=total_succ,
        total_failures=total_fail,
        target_recall=target_recall,
    )


class BaseProbe(ABC):
    """Abstract Base Class for all Probes."""

    def __init__(self, name: str, C: float = 1.0, random_state: int = 42):
        self.name = name
        self.C = C
        self.random_state = random_state

    @abstractmethod
    def fit(self, episodes: list[dict[str, Any]]) -> BaseProbe:
        """Fits probe models on a list of structured episode objects."""
        pass

    @abstractmethod
    def score_step(self, episode: dict[str, Any], round_idx: int) -> Optional[float]:
        """Scores an individual turn within an episode. Returns probability or decision value."""
        pass

    @abstractmethod
    def evaluate_cascade(
        self,
        episodes: list[dict[str, Any]],
        target_recall: float = 0.95,
    ) -> CascadeMetrics:
        """Evaluates early-abort cascade gating under Clopper-Pearson recall control."""
        pass

    @abstractmethod
    def extract_features(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts 2D feature matrix X and label vector y for static classification."""
        pass
