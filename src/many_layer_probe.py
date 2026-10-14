"""
6. Many Layer Input Probe (Multi-Layer Residual Fusion Probe):
Concatenates residual-stream representations across multiple transformer layers:
h_many = [h^(l_1), h^(l_2), ..., h^(l_K)]
Evaluates whether hierarchical joint representations (e.g. shallow syntax + middle planning + deep emission)
improve failure detection and cross-domain transfer over single-layer probes.
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


class ManyLayerProbe(BaseProbe):
    """
    Many Layer Input Probe.
    Concatenates hidden states across multiple layers: [h^(l_1), ..., h^(l_K)].
    """

    def __init__(
        self,
        layers: Optional[list[int]] = None,
        anchor: str = "post_gen",
        C: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__(name="Many Layer Input Probe", C=C, random_state=random_state)
        # Default to standard cross-depth layers [10, 20, 27] or all available
        self.layers = layers or [4, 10, 16, 18, 20, 22, 24, 27]
        self.anchor = anchor
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state),
        )
        self.threshold: float = 0.5

    def _get_fused_vector(self, episode: dict[str, Any], round_idx: int) -> Optional[np.ndarray]:
        rounds = episode.get("rounds", {})
        if round_idx not in rounds or self.anchor not in rounds[round_idx]:
            return None

        layer_dict = rounds[round_idx][self.anchor]
        available_layers = [l for l in self.layers if l in layer_dict]
        if not available_layers:
            return None

        fused = np.concatenate([layer_dict[l] for l in available_layers])
        return fused.astype(np.float32)

    def extract_features(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for ep in episodes:
            for r in sorted(ep["rounds"].keys()):
                vec = self._get_fused_vector(ep, r)
                if vec is not None:
                    X_list.append(vec)
                    y_list.append(ep["success"])

        if not X_list:
            raise ValueError(f"No multi-layer features found for layers={self.layers}")

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)

    def fit(self, episodes: list[dict[str, Any]]) -> ManyLayerProbe:
        X, y = self.extract_features(episodes)
        self.pipeline.fit(X, y)

        succ_scores = []
        for ep in episodes:
            if ep["success"] == 1:
                for r in sorted(ep["rounds"].keys()):
                    vec = self._get_fused_vector(ep, r)
                    if vec is not None:
                        prob = float(self.pipeline.predict_proba(vec.reshape(1, -1))[0, 1])
                        succ_scores.append(prob)

        if succ_scores:
            self.threshold = float(np.percentile(succ_scores, 5.0))
        else:
            self.threshold = 0.5

        return self

    def score_step(self, episode: dict[str, Any], round_idx: int) -> Optional[float]:
        vec = self._get_fused_vector(episode, round_idx)
        if vec is not None:
            return float(self.pipeline.predict_proba(vec.reshape(1, -1))[0, 1])
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

            fold_probe = ManyLayerProbe(layers=self.layers, anchor=self.anchor, C=self.C, random_state=self.random_state)
            fold_probe.fit(train_eps)

            X_val, y_val = fold_probe.extract_features(val_eps)
            probs = fold_probe.pipeline.predict_proba(X_val)[:, 1]

            all_y_true.extend(y_val)
            all_y_prob.extend(probs)

        metrics = evaluate_binary_metrics(np.array(all_y_true), np.array(all_y_prob))
        return metrics, metrics.roc_auc
