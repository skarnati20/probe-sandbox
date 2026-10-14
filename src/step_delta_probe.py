"""
3. Step Delta Probe:
Measures inter-round vector difference with the previous hidden state:
Delta h_step,t = h_t^post - h_{t-1}^post (0 at round 0).
Tests whether latent trajectory step velocity/displacement encodes task failure or progress.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import (
    BaseProbe,
    CascadeMetrics,
    ProbeMetrics,
    clopper_pearson_lower,
    evaluate_abort_policy,
    evaluate_binary_metrics,
)


class StepDeltaProbe(BaseProbe):
    """
    Step Delta Probe.
    Classifies success/failure based on the inter-round displacement vector:
    Delta h_t = h_t^post - h_{t-1}^post.
    """

    def __init__(
        self,
        layer: int = 20,
        anchor: str = "post_gen",
        C: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__(name="Step Delta Probe", C=C, random_state=random_state)
        self.layer = layer
        self.anchor = anchor
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state),
        )
        self.threshold: float = 0.5

    def _get_step_delta(self, episode: dict[str, Any], round_idx: int) -> Optional[np.ndarray]:
        """Calculates Delta h_t = h_t - h_{t-1}."""
        rounds = episode.get("rounds", {})
        if round_idx not in rounds or self.anchor not in rounds[round_idx]:
            return None
        if self.layer not in rounds[round_idx][self.anchor]:
            return None

        h_curr = rounds[round_idx][self.anchor][self.layer]

        if round_idx > 0 and (round_idx - 1) in rounds and self.anchor in rounds[round_idx - 1]:
            if self.layer in rounds[round_idx - 1][self.anchor]:
                h_prev = rounds[round_idx - 1][self.anchor][self.layer]
                return (h_curr - h_prev).astype(np.float32)

        # Round 0: zero displacement vector (baseline state)
        return np.zeros_like(h_curr, dtype=np.float32)

    def extract_features(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for ep in episodes:
            for r in sorted(ep["rounds"].keys()):
                delta = self._get_step_delta(ep, r)
                if delta is not None:
                    X_list.append(delta)
                    y_list.append(ep["success"])

        if not X_list:
            raise ValueError(f"No step delta features found for layer={self.layer}")

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)

    def fit(self, episodes: list[dict[str, Any]]) -> StepDeltaProbe:
        X, y = self.extract_features(episodes)
        self.pipeline.fit(X, y)

        succ_scores = []
        for ep in episodes:
            if ep["success"] == 1:
                for r in sorted(ep["rounds"].keys()):
                    delta = self._get_step_delta(ep, r)
                    if delta is not None:
                        prob = float(self.pipeline.predict_proba(delta.reshape(1, -1))[0, 1])
                        succ_scores.append(prob)

        if succ_scores:
            self.threshold = float(np.percentile(succ_scores, 5.0))
        else:
            self.threshold = 0.5

        return self

    def score_step(self, episode: dict[str, Any], round_idx: int) -> Optional[float]:
        delta = self._get_step_delta(episode, round_idx)
        if delta is not None:
            return float(self.pipeline.predict_proba(delta.reshape(1, -1))[0, 1])
        return None

    def evaluate_classification(self, episodes: list[dict[str, Any]]) -> ProbeMetrics:
        X, y = self.extract_features(episodes)
        probs = self.pipeline.predict_proba(X)[:, 1]
        return evaluate_binary_metrics(y, probs)

    def evaluate_cascade(
        self,
        episodes: list[dict[str, Any]],
        target_recall: float = 0.95,
    ) -> CascadeMetrics:
        return evaluate_abort_policy(
            episodes,
            score_fn=self.score_step,
            threshold_fn=lambda r: self.threshold,
            target_recall=target_recall,
        )

    def evaluate_cv(
        self,
        episodes: list[dict[str, Any]],
        n_splits: int = 5,
    ) -> tuple[ProbeMetrics, float]:
        y_ep = np.array([ep["success"] for ep in episodes])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        all_y_true = []
        all_y_prob = []

        for train_idx, val_idx in skf.split(episodes, y_ep):
            train_eps = [episodes[i] for i in train_idx]
            val_eps = [episodes[i] for i in val_idx]

            fold_probe = StepDeltaProbe(layer=self.layer, anchor=self.anchor, C=self.C, random_state=self.random_state)
            fold_probe.fit(train_eps)

            X_val, y_val = fold_probe.extract_features(val_eps)
            probs = fold_probe.pipeline.predict_proba(X_val)[:, 1]

            all_y_true.extend(y_val)
            all_y_prob.extend(probs)

        metrics = evaluate_binary_metrics(np.array(all_y_true), np.array(all_y_prob))
        return metrics, metrics.roc_auc
