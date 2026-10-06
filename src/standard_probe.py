from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class StandardProbeMetrics:
    roc_auc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int
    positive_class_ratio: float


class StandardActivationProbe:
    """
    Standard linear / MLP probe trained on LLM residual stream activations.
    Evaluates whether internal hidden states linearly encode episode success vs failure.
    """

    def __init__(
        self,
        classifier_type: str = "logistic",
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.classifier_type = classifier_type.lower()
        self.C = C
        self.random_state = random_state
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        if self.classifier_type == "logistic":
            clf = LogisticRegression(
                C=self.C,
                max_iter=2000,
                random_state=self.random_state,
            )
        elif self.classifier_type == "ridge":
            clf = RidgeClassifier(
                alpha=1.0 / self.C,
                random_state=self.random_state,
            )
        elif self.classifier_type == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(256,),
                alpha=1e-3,
                max_iter=500,
                early_stopping=True,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        return make_pipeline(StandardScaler(), clf)

    def fit(self, X: np.ndarray, y: np.ndarray) -> StandardActivationProbe:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.pipeline.named_steps[self.pipeline.steps[-1][0]], "predict_proba"):
            return self.pipeline.predict_proba(X)[:, 1]
        elif hasattr(self.pipeline, "decision_function"):
            df = self.pipeline.decision_function(X)
            return 1.0 / (1.0 + np.exp(-df))
        return self.pipeline.predict(X).astype(float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> tuple[StandardProbeMetrics, np.ndarray]:
        """
        Runs Stratified K-Fold Cross-Validation and computes out-of-fold metrics.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in skf.split(X, y):
            fold_pipe = self._build_pipeline()
            fold_pipe.fit(X[train_idx], y[train_idx])

            if hasattr(fold_pipe.named_steps[fold_pipe.steps[-1][0]], "predict_proba"):
                oof_preds[val_idx] = fold_pipe.predict_proba(X[val_idx])[:, 1]
            else:
                df = fold_pipe.decision_function(X[val_idx])
                oof_preds[val_idx] = 1.0 / (1.0 + np.exp(-df))

        # Check for binary variety
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, oof_preds))
        else:
            auc = 0.5

        binary_preds = (oof_preds >= 0.5).astype(int)
        metrics = StandardProbeMetrics(
            roc_auc=auc,
            accuracy=float(accuracy_score(y, binary_preds)),
            balanced_accuracy=float(balanced_accuracy_score(y, binary_preds)),
            precision=float(precision_score(y, binary_preds, zero_division=0)),
            recall=float(recall_score(y, binary_preds, zero_division=0)),
            f1=float(f1_score(y, binary_preds, zero_division=0)),
            num_samples=len(y),
            positive_class_ratio=float(np.mean(y)),
        )

        return metrics, oof_preds
