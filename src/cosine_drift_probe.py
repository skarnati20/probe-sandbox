"""
5. Cosine Drift Probe:
Measures directional drift and angular kinematics across trajectory states:
- Cosine similarity with previous step: cos(h_t, h_{t-1})
- Cosine similarity with initial state: cos(h_t, h_0)
- Running trajectory path tortuosity and directional alignment statistics.
Decodes whether the agent is pursuing a straight, directed path or aimlessly wandering / looping.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
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


class CosineDriftProbe(BaseProbe):
    """
    Cosine Drift Probe.
    Classifies trajectories based on angular drift, path tortuosity, and cosine displacement metrics.
    """

    def __init__(
        self,
        layer: int = 20,
        anchor: str = "post_gen",
        C: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__(name="Cosine Drift Probe", C=C, random_state=random_state)
        self.layer = layer
        self.anchor = anchor
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state),
        )
        self.threshold: float = 0.5

    def _extract_drift_features(self, episode: dict[str, Any], current_round: int) -> Optional[np.ndarray]:
        """Computes 8-D cosine drift & kinematic metrics up to current_round."""
        rounds = sorted(r for r in episode.get("rounds", {}).keys() if r <= current_round)
        if not rounds:
            return None

        states = []
        for r in rounds:
            r_data = episode["rounds"][r]
            if self.anchor in r_data and self.layer in r_data[self.anchor]:
                states.append(r_data[self.anchor][self.layer])

        if not states:
            return None

        h0 = states[0]
        norm_h0 = float(np.linalg.norm(h0)) + 1e-8
        h_curr = states[-1]

        if len(states) == 1:
            # Step 0 baseline: no previous displacement
            return np.array([
                1.0,   # cos with prev (identity)
                1.0,   # cos with origin
                0.0,   # step displacement
                0.0,   # origin displacement
                1.0,   # path efficiency
                1.0,   # mean step cos
                1.0,   # min step cos
                0.0,   # total path
            ], dtype=np.float32)

        # Step cosine similarities
        step_cosines = [
            float(cosine_similarity(states[i - 1].reshape(1, -1), states[i].reshape(1, -1))[0, 0])
            for i in range(1, len(states))
        ]

        # Origin cosine similarity
        orig_cos = float(cosine_similarity(h0.reshape(1, -1), h_curr.reshape(1, -1))[0, 0])

        # Step displacement
        h_prev = states[-2]
        step_disp = float(np.linalg.norm(h_curr - h_prev)) / norm_h0

        # Origin net displacement
        orig_disp = float(np.linalg.norm(h_curr - h0)) / norm_h0

        # Cumulative path length
        total_path = sum(float(np.linalg.norm(states[i] - states[i - 1])) / norm_h0 for i in range(1, len(states)))

        # Path efficiency (straightness / tortuosity)
        efficiency = orig_disp / (total_path + 1e-8)

        mean_step_cos = float(np.mean(step_cosines))
        min_step_cos = float(np.min(step_cosines))

        return np.array([
            step_cosines[-1],
            orig_cos,
            step_disp,
            orig_disp,
            efficiency,
            mean_step_cos,
            min_step_cos,
            total_path,
        ], dtype=np.float32)

    def extract_features(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for ep in episodes:
            for r in sorted(ep["rounds"].keys()):
                feat = self._extract_drift_features(ep, r)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(ep["success"])

        if not X_list:
            raise ValueError("No cosine drift features found.")

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)

    def fit(self, episodes: list[dict[str, Any]]) -> CosineDriftProbe:
        X, y = self.extract_features(episodes)
        self.pipeline.fit(X, y)

        succ_scores = []
        for ep in episodes:
            if ep["success"] == 1:
                for r in sorted(ep["rounds"].keys()):
                    feat = self._extract_drift_features(ep, r)
                    if feat is not None:
                        prob = float(self.pipeline.predict_proba(feat.reshape(1, -1))[0, 1])
                        succ_scores.append(prob)

        if succ_scores:
            self.threshold = float(np.percentile(succ_scores, 5.0))
        else:
            self.threshold = 0.5

        return self

    def score_step(self, episode: dict[str, Any], round_idx: int) -> Optional[float]:
        feat = self._extract_drift_features(episode, round_idx)
        if feat is not None:
            return float(self.pipeline.predict_proba(feat.reshape(1, -1))[0, 1])
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

            fold_probe = CosineDriftProbe(layer=self.layer, anchor=self.anchor, C=self.C, random_state=self.random_state)
            fold_probe.fit(train_eps)

            X_val, y_val = fold_probe.extract_features(val_eps)
            probs = fold_probe.pipeline.predict_proba(X_val)[:, 1]

            all_y_true.extend(y_val)
            all_y_prob.extend(probs)

        metrics = evaluate_binary_metrics(np.array(all_y_true), np.array(all_y_prob))
        return metrics, metrics.roc_auc
