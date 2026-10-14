"""
Unit tests for the 6 refactored probes:
1. Standard Probe
2. Doomed Probe (Paper Cascade)
3. Step Delta Probe
4. Token Delta Probe (Post - Pre)
5. Cosine Drift Probe
6. Many Layer Input Probe
"""

from pathlib import Path
import unittest

import numpy as np

from src.base import clopper_pearson_lower
from src.cosine_drift_probe import CosineDriftProbe
from src.data import ShardDataset, extract_episodes
from src.doomed_probe import DoomedProbe
from src.many_layer_probe import ManyLayerProbe
from src.standard_probe import StandardProbe
from src.step_delta_probe import StepDeltaProbe
from src.token_delta_probe import TokenDeltaProbe


class TestSixProbesPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_episodes = 24
        self.turns_per_ep = 4
        self.layers = [10, 20]
        self.hidden_dim = 16

        self.metas = []
        feats_list = []
        self.y_list = []
        self.task_idx_list = []
        self.rollout_k_list = []
        self.rounds_list = []
        self.anchors_list = []

        u = np.ones((len(self.layers), self.hidden_dim), dtype=np.float32) / np.sqrt(self.hidden_dim)

        for ep_i in range(self.num_episodes):
            succ = int(ep_i % 2 == 0)  # Balanced 50/50
            task_id = ep_i // 4
            rollout_k = ep_i % 4
            ep_name = f"task_{task_id}_k{rollout_k}"

            curr_state = np.zeros((len(self.layers), self.hidden_dim), dtype=np.float32)

            for t in range(self.turns_per_ep):
                for anchor in ["post_gen", "pre_gen"]:
                    if succ:
                        step_disp = u * 1.0 + np.random.randn(len(self.layers), self.hidden_dim).astype(np.float32) * 0.05
                    else:
                        step_disp = np.random.randn(len(self.layers), self.hidden_dim).astype(np.float32) * 1.5

                    if anchor == "pre_gen":
                        feat = curr_state + step_disp * 0.5
                    else:
                        curr_state = curr_state + step_disp
                        feat = curr_state.copy()

                    feats_list.append(feat)
                    self.y_list.append(succ)
                    self.task_idx_list.append(task_id)
                    self.rollout_k_list.append(rollout_k)
                    self.rounds_list.append(t)
                    self.anchors_list.append(anchor)

                    self.metas.append({
                        "task_idx": task_id,
                        "rollout_k": rollout_k,
                        "round": t,
                        "anchor": anchor,
                        "success": succ,
                        "episode_id": ep_name,
                    })

        self.dataset = ShardDataset(
            X=np.stack(feats_list, axis=0),
            y=np.array(self.y_list, dtype=np.int32),
            task_idx=np.array(self.task_idx_list, dtype=np.int32),
            rollout_k=np.array(self.rollout_k_list, dtype=np.int32),
            rounds=np.array(self.rounds_list, dtype=np.int32),
            anchors=np.array(self.anchors_list),
            layer_ids=self.layers,
            metas=self.metas,
        )
        self.episodes = extract_episodes(self.dataset)

    def test_clopper_pearson(self):
        lb = clopper_pearson_lower(k=95, n=100, confidence=0.95)
        self.assertGreater(lb, 0.85)
        self.assertLessEqual(lb, 0.95)

    def test_probe_1_standard(self):
        probe = StandardProbe(layer=20, anchor="post_gen")
        probe.fit(self.episodes)
        metrics = probe.evaluate_classification(self.episodes)
        self.assertGreaterEqual(metrics.roc_auc, 0.5)

        casc = probe.evaluate_cascade(self.episodes, target_recall=0.90)
        self.assertGreaterEqual(casc.recall_preserved, 0.0)

        cv_metrics, auc = probe.evaluate_cv(self.episodes, n_splits=3)
        self.assertGreaterEqual(auc, 0.5)

    def test_probe_2_doomed(self):
        probe = DoomedProbe(layer=20, anchor="post_gen")
        probe.fit(self.episodes, target_recall=0.90)
        casc = probe.evaluate_cascade(self.episodes, target_recall=0.90)
        self.assertGreaterEqual(casc.recall_preserved, 0.0)

        cv_metrics, avg_casc = probe.evaluate_cv(self.episodes, n_splits=3, target_recall=0.90)
        self.assertGreaterEqual(cv_metrics.roc_auc, 0.5)

    def test_probe_3_step_delta(self):
        probe = StepDeltaProbe(layer=20, anchor="post_gen")
        probe.fit(self.episodes)
        metrics = probe.evaluate_classification(self.episodes)
        self.assertGreaterEqual(metrics.roc_auc, 0.5)

        cv_metrics, auc = probe.evaluate_cv(self.episodes, n_splits=3)
        self.assertGreaterEqual(auc, 0.5)

    def test_probe_4_token_delta(self):
        probe = TokenDeltaProbe(layer=20)
        probe.fit(self.episodes)
        metrics = probe.evaluate_classification(self.episodes)
        self.assertGreaterEqual(metrics.roc_auc, 0.5)

        cv_metrics, auc = probe.evaluate_cv(self.episodes, n_splits=3)
        self.assertGreaterEqual(auc, 0.5)

    def test_probe_5_cosine_drift(self):
        probe = CosineDriftProbe(layer=20, anchor="post_gen")
        probe.fit(self.episodes)
        metrics = probe.evaluate_classification(self.episodes)
        self.assertGreaterEqual(metrics.roc_auc, 0.5)

        cv_metrics, auc = probe.evaluate_cv(self.episodes, n_splits=3)
        self.assertGreaterEqual(auc, 0.5)

    def test_probe_6_many_layer(self):
        probe = ManyLayerProbe(layers=[10, 20], anchor="post_gen")
        probe.fit(self.episodes)
        metrics = probe.evaluate_classification(self.episodes)
        self.assertGreaterEqual(metrics.roc_auc, 0.5)

        cv_metrics, auc = probe.evaluate_cv(self.episodes, n_splits=3)
        self.assertGreaterEqual(auc, 0.5)


if __name__ == "__main__":
    unittest.main()
