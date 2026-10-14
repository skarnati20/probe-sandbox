"""
2. Doomed Probe (Paper Cascade Probe):
Round-wise early-abort cascade from 'Doomed from the Start' (arXiv:2607.06503).
Trains separate logistic gates at each round t on h_t^post with Clopper-Pearson recall guarantees.
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


class DoomedProbe(BaseProbe):
    """
    Paper Cascade Probe ('Doomed Probe').
    Maintains a sequence of round-specific classifiers {W_0, W_1, ..., W_T} calibrated with
    Clopper-Pearson binomial confidence bounds to abort doomed trajectories as early as possible.
    """

    def __init__(
        self,
        layer: int = 20,
        anchor: str = "post_gen",
        C: float = 1.0,
        confidence: float = 0.95,
        random_state: int = 42,
    ):
        super().__init__(name="Doomed Probe (Paper Cascade)", C=C, random_state=random_state)
        self.layer = layer
        self.anchor = anchor
        self.confidence = confidence

        # Round-indexed models and calibrated thresholds
        self.round_models: dict[int, Any] = {}
        self.round_thresholds: dict[int, float] = {}

    def extract_features(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts all round states as a pooled matrix for baseline comparison."""
        X_list, y_list = [], []
        for ep in episodes:
            for r_data in ep["rounds"].values():
                if self.anchor in r_data and self.layer in r_data[self.anchor]:
                    X_list.append(r_data[self.anchor][self.layer])
                    y_list.append(ep["success"])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)

    def fit(
        self,
        episodes: list[dict[str, Any]],
        target_recall: float = 0.95,
    ) -> DoomedProbe:
        max_round = max(max(ep["rounds"].keys()) for ep in episodes if ep["rounds"])
        self.round_models = {}
        self.round_thresholds = {}

        for r in range(max_round + 1):
            X_r, y_r = [], []
            for ep in episodes:
                if r in ep["rounds"] and self.anchor in ep["rounds"][r]:
                    layer_dict = ep["rounds"][r][self.anchor]
                    if self.layer in layer_dict:
                        X_r.append(layer_dict[self.layer])
                        y_r.append(ep["success"])

            if len(y_r) >= 6 and len(set(y_r)) > 1:
                pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state),
                )
                pipe.fit(np.array(X_r), np.array(y_r))
                self.round_models[r] = pipe

                # Calibrate Clopper-Pearson threshold on training successes
                succ_acts = [
                    ep["rounds"][r][self.anchor][self.layer]
                    for ep in episodes
                    if ep["success"] == 1
                    and r in ep["rounds"]
                    and self.anchor in ep["rounds"][r]
                    and self.layer in ep["rounds"][r][self.anchor]
                ]

                if succ_acts:
                    succ_probs = pipe.predict_proba(np.array(succ_acts))[:, 1]
                    sorted_probs = np.sort(succ_probs)
                    n = len(sorted_probs)
                    best_th = sorted_probs[0]

                    for k in range(1, n + 1):
                        cand_th = sorted_probs[n - k]
                        if clopper_pearson_lower(k, n, confidence=self.confidence) >= target_recall:
                            best_th = cand_th
                            break
                    self.round_thresholds[r] = float(best_th)

        return self

    def score_step(self, episode: dict[str, Any], round_idx: int) -> Optional[float]:
        if not self.round_models:
            return None

        # Fallback to nearest/latest round model if test episode has more steps than training
        r_key = round_idx if round_idx in self.round_models else max(self.round_models.keys())

        rounds = episode.get("rounds", {})
        if round_idx in rounds and self.anchor in rounds[round_idx]:
            layer_dict = rounds[round_idx][self.anchor]
            if self.layer in layer_dict:
                vec = layer_dict[self.layer].reshape(1, -1)
                return float(self.round_models[r_key].predict_proba(vec)[0, 1])
        return None

    def evaluate_cascade(
        self,
        episodes: list[dict[str, Any]],
        target_recall: float = 0.95,
    ) -> CascadeMetrics:
        def get_thresh(r):
            if r in self.round_thresholds:
                return self.round_thresholds[r]
            elif self.round_thresholds:
                return self.round_thresholds[max(self.round_thresholds.keys())]
            return None

        return evaluate_abort_policy(
            episodes,
            score_fn=self.score_step,
            threshold_fn=get_thresh,
            target_recall=target_recall,
        )

    def evaluate_classification(self, episodes: list[dict[str, Any]]) -> ProbeMetrics:
        """Evaluates round-weighted classification accuracy."""
        all_y_true, all_y_prob = [], []
        for ep in episodes:
            for r in sorted(ep["rounds"].keys()):
                score = self.score_step(ep, r)
                if score is not None:
                    all_y_true.append(ep["success"])
                    all_y_prob.append(score)

        return evaluate_binary_metrics(np.array(all_y_true), np.array(all_y_prob))

    def evaluate_cv(
        self,
        episodes: list[dict[str, Any]],
        n_splits: int = 5,
        target_recall: float = 0.95,
    ) -> tuple[ProbeMetrics, CascadeMetrics]:
        y_ep = np.array([ep["success"] for ep in episodes])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        all_y_true, all_y_prob = [], []
        fold_cascades = []

        for train_idx, val_idx in skf.split(episodes, y_ep):
            train_eps = [episodes[i] for i in train_idx]
            val_eps = [episodes[i] for i in val_idx]

            fold_probe = DoomedProbe(
                layer=self.layer,
                anchor=self.anchor,
                C=self.C,
                confidence=self.confidence,
                random_state=self.random_state,
            )
            fold_probe.fit(train_eps, target_recall=target_recall)

            casc_res = fold_probe.evaluate_cascade(val_eps, target_recall=target_recall)
            fold_cascades.append(casc_res)

            for ep in val_eps:
                for r in sorted(ep["rounds"].keys()):
                    score = fold_probe.score_step(ep, r)
                    if score is not None:
                        all_y_true.append(ep["success"])
                        all_y_prob.append(score)

        metrics = evaluate_binary_metrics(np.array(all_y_true), np.array(all_y_prob))

        # Average cascade metrics across folds
        avg_casc = CascadeMetrics(
            recall_preserved=float(np.mean([c.recall_preserved for c in fold_cascades])),
            false_abort_rate=float(np.mean([c.false_abort_rate for c in fold_cascades])),
            doomed_abort_rate=float(np.mean([c.doomed_abort_rate for c in fold_cascades])),
            step_savings=float(np.mean([c.step_savings for c in fold_cascades])),
            avg_abort_step=float(np.mean([c.avg_abort_step for c in fold_cascades])),
            total_episodes=len(episodes),
            total_successes=int(np.sum(y_ep == 1)),
            total_failures=int(np.sum(y_ep == 0)),
            target_recall=target_recall,
        )
        return metrics, avg_casc
