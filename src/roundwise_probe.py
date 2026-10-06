from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ShardDataset


@dataclass
class RoundProbeResult:
    round_idx: int
    auc: float
    num_samples: int
    num_success: int
    num_fail: int
    oof_scores: np.ndarray


def fit_oof_logistic(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Computes out-of-fold decision scores using GroupKFold(groups=task_idx).
    Mirrors 'oof_scores' from the paper.
    """
    scores = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)

    if actual_splits < 2:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        pipe.fit(X, y)
        return pipe.decision_function(X)

    gkf = GroupKFold(n_splits=actual_splits)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
        clf.fit(sc.transform(X[tr]), y[tr])
        scores[te] = clf.decision_function(sc.transform(X[te]))

    assert not np.isnan(scores).any(), "NaN found in out-of-fold scores"
    return scores


def within_task_pair_acc(
    scores: np.ndarray,
    y: np.ndarray,
    task_ids: np.ndarray,
) -> float:
    """
    Evaluates pairwise ranking accuracy on rollouts of the same task.
    Mirrors 'within_task_pair_acc' from the paper.
    """
    by_task = defaultdict(list)
    for s, yy, t in zip(scores, y, task_ids):
        by_task[t].append((s, yy))

    wins, ties, total = 0, 0, 0
    for t, items in by_task.items():
        pos = [s for s, yy in items if yy == 1]
        neg = [s for s, yy in items if yy == 0]
        if not pos or not neg:
            continue
        for p in pos:
            for n in neg:
                total += 1
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


class RoundwiseProber:
    """
    Round-wise residual stream scorer using GroupKFold cross-validation on task_idx.
    Measures the emergence of failure predictability over time (AUC-over-round).
    """

    def __init__(
        self,
        dataset: ShardDataset,
        C: float = 1.0,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.C = C
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate_round(
        self,
        round_idx: int,
        anchor: str = "post_gen",
        layer_idx: Optional[int] = None,
    ) -> RoundProbeResult:
        """
        Trains and evaluates a GroupKFold probe at a specific turn index t.
        """
        X_t, y_t, groups_t = self.dataset.get_slice(
            anchor=anchor,
            round_idx=round_idx,
            layer_idx=layer_idx,
        )

        n_pos = int(np.sum(y_t == 1))
        n_neg = int(np.sum(y_t == 0))

        if n_pos == 0 or n_neg == 0:
            return RoundProbeResult(
                round_idx=round_idx,
                auc=0.5,
                num_samples=len(y_t),
                num_success=n_pos,
                num_fail=n_neg,
                oof_scores=np.zeros(len(y_t)),
            )

        oof_scores = fit_oof_logistic(
            X_t,
            y_t,
            groups=groups_t,
            n_splits=self.n_splits,
            C=self.C,
            random_state=self.random_state,
        )

        auc = float(roc_auc_score(y_t, oof_scores))

        return RoundProbeResult(
            round_idx=round_idx,
            auc=auc,
            num_samples=len(y_t),
            num_success=n_pos,
            num_fail=n_neg,
            oof_scores=oof_scores,
        )

    def evaluate_auc_over_rounds(
        self,
        anchor: str = "post_gen",
        layer_idx: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> list[RoundProbeResult]:
        """
        Computes the complete AUC-over-round curve across all rounds t=0..T-1.
        (Reproduces Figure 3 and Figure 4 from the paper).
        """
        avail_rounds = sorted(np.unique(self.dataset.rounds))
        if max_rounds is not None:
            avail_rounds = [r for r in avail_rounds if r < max_rounds]

        results = []
        for t in avail_rounds:
            try:
                res = self.evaluate_round(round_idx=t, anchor=anchor, layer_idx=layer_idx)
                results.append(res)
            except ValueError:
                continue

        return results

    def layer_sweep(
        self,
        round_idx: int = 0,
        anchor: str = "post_gen",
    ) -> dict[int, float]:
        """
        Evaluates probe AUC across every extracted layer (Supplementary A layer sweep).
        """
        layer_aucs = {}
        for l in self.dataset.layer_ids:
            res = self.evaluate_round(round_idx=round_idx, anchor=anchor, layer_idx=l)
            layer_aucs[l] = res.auc
        return layer_aucs


RoundwiseScorer = RoundwiseProber
