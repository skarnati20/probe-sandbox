"""
1. Standard Activation Probe:
Single-hyperplane linear / logistic probe trained on static residual stream activations (h_t^post).
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
    evaluate_abort_policy,
    evaluate_binary_metrics,
)


class StandardProbe(BaseProbe):
    """
    Standard static activation probe.
    Fits a global linear decision hyperplane across static hidden states h_t.
    """

    def __init__(
        self,
        layer: int = 20,
        anchor: str = "post_gen",
        C: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__(name="Standard Probe", C=C, random_state=random_state)
        self.layer = layer
        self.anchor = anchor
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state),
        )
        self.threshold: float = 0.5

    def extract_features(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for ep in episodes:
            for r_data in ep["rounds"].values():
                if self.anchor in r_data and self.layer in r_data[self.anchor]:
                    X_list.append(r_data[self.anchor][self.layer])
                    y_list.append(ep["success"])

        if not X_list:
            raise ValueError(f"No features found for anchor='{self.anchor}', layer={self.layer}")

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)

    def fit(self, episodes: list[dict[str, Any]]) -> StandardProbe:
        X, y = self.extract_features(episodes)
        self.pipeline.fit(X, y)

        # Compute default recall threshold for cascade simulation
        succ_scores = []
        for ep in episodes:
            if ep["success"] == 1:
                for r_data in ep["rounds"].values():
                    if self.anchor in r_data and self.layer in r_data[self.anchor]:
                        vec = r_data[self.anchor][self.layer].reshape(1, -1)
                        prob = float(self.pipeline.predict_proba(vec)[0, 1])
                        succ_scores.append(prob)

        if succ_scores:
            self.threshold = float(np.percentile(succ_scores, 5.0))  # default 95% recall threshold
        else:
            self.threshold = 0.5

        return self

    def score_step(self, episode: dict[str, Any], round_idx: int) -> Optional[float]:
        rounds = episode.get("rounds", {})
        if round_idx in rounds and self.anchor in rounds[round_idx]:
            layer_dict = rounds[round_idx][self.anchor]
            if self.layer in layer_dict:
                vec = layer_dict[self.layer].reshape(1, -1)
                return float(self.pipeline.predict_proba(vec)[0, 1])
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
        # Calibrate threshold to target_recall
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
        """Runs Stratified K-Fold CV across episodes."""
        y_ep = np.array([ep["success"] for ep in episodes])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        all_y_true = []
        all_y_prob = []

        for train_idx, val_idx in skf.split(episodes, y_ep):
            train_eps = [episodes[i] for i in train_idx]
            val_eps = [episodes[i] for i in val_idx]

            fold_probe = StandardProbe(layer=self.layer, anchor=self.anchor, C=self.C, random_state=self.random_state)
            fold_probe.fit(train_eps)

            X_val, y_val = fold_probe.extract_features(val_eps)
            probs = fold_probe.pipeline.predict_proba(X_val)[:, 1]

            all_y_true.extend(y_val)
            all_y_prob.extend(probs)

        metrics = evaluate_binary_metrics(np.array(all_y_true), np.array(all_y_prob))
        return metrics, metrics.roc_auc
