"""
Unit tests for the probing and cascade evaluation pipeline.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.cascade import RecallControlledCascade, clopper_pearson_lower
from src.data import ShardDataset, load_shards
from src.roundwise_probe import RoundwiseProber, fit_oof_logistic, within_task_pair_acc
from src.standard_probe import StandardActivationProbe


class TestProbingPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_samples = 100
        self.num_layers = 2
        self.hidden_dim = 32

        # Create synthetic dataset with slight class separation
        self.y = np.random.binomial(1, 0.4, size=self.num_samples)
        self.task_idx = np.repeat(np.arange(20), 5)  # 20 tasks, 5 rollouts each = 100 episodes
        self.rollout_k = np.tile(np.arange(5), 20)
        self.rounds = np.random.randint(0, 4, size=self.num_samples)
        self.anchors = np.random.choice(["post_gen", "pre_gen"], size=self.num_samples)

        # Signal in activation based on y
        signal = self.y[:, None, None] * 0.5
        noise = np.random.randn(self.num_samples, self.num_layers, self.hidden_dim) * 0.5
        self.X = (signal + noise).astype(np.float32)

        self.metas = [
            {
                "task_idx": int(self.task_idx[i]),
                "rollout_k": int(self.rollout_k[i]),
                "round": int(self.rounds[i]),
                "anchor": str(self.anchors[i]),
                "success": int(self.y[i]),
                "episode_id": f"task_{self.task_idx[i]}_k{self.rollout_k[i]}",
                "n_rounds": 4,
            }
            for i in range(self.num_samples)
        ]

        self.dataset = ShardDataset(
            X=self.X,
            y=self.y,
            task_idx=self.task_idx,
            rollout_k=self.rollout_k,
            rounds=self.rounds,
            anchors=self.anchors,
            layer_ids=[14, 20],
            metas=self.metas,
        )

    def test_standard_probe(self):
        X_2d, y, _ = self.dataset.get_slice(anchor=None)
        probe = StandardActivationProbe(classifier_type="logistic", C=1.0)
        metrics, oof_preds = probe.evaluate_cv(X_2d, y, n_splits=3)

        self.assertGreaterEqual(metrics.roc_auc, 0.5)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertEqual(len(oof_preds), len(y))

    def test_roundwise_prober(self):
        prober = RoundwiseProber(dataset=self.dataset, C=1.0, n_splits=3)
        round_res = prober.evaluate_round(round_idx=0, anchor="post_gen")
        self.assertIsInstance(round_res.auc, float)

        pair_acc = within_task_pair_acc(
            round_res.oof_scores,
            self.y[: len(round_res.oof_scores)],
            self.task_idx[: len(round_res.oof_scores)],
        )
        self.assertGreaterEqual(pair_acc, 0.0)

    def test_clopper_pearson(self):
        # 95 out of 100 successes should have lower bound > 0.85
        lb = clopper_pearson_lower(k=95, n=100, confidence=0.95)
        self.assertGreater(lb, 0.85)
        self.assertLessEqual(lb, 0.95)

    def test_cascade_evaluation(self):
        cascade = RecallControlledCascade(confidence=0.95, n_splits=3, C=1.0)
        res = cascade.evaluate_cascade(self.dataset, target_recall=0.90, anchor="post_gen")

        self.assertGreaterEqual(res.actual_recall, 0.0)
        self.assertGreaterEqual(res.step_savings_rate, 0.0)
        self.assertEqual(res.total_episodes, 100)  # 100 unique episodes


if __name__ == "__main__":
    unittest.main()
