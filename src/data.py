from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ShardDataset:
    """
    In-memory representation of extracted activation shards and matching metadata.
    """
    X: np.ndarray             # Shape: [N, n_layers, d_model] in float32 or float16
    y: np.ndarray             # Binary success labels [N] (0 or 1)
    task_idx: np.ndarray      # Grouping IDs [N] for GroupKFold
    rollout_k: np.ndarray     # Rollout index [N]
    rounds: np.ndarray        # Round indices [N] (0..T-1)
    anchors: np.ndarray       # Anchor names [N] ('post_gen', 'pre_gen', etc.)
    layer_ids: list[int]      # Layer indices corresponding to dim 1 of X
    metas: list[dict[str, Any]]

    @property
    def num_samples(self) -> int:
        return self.X.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.X.shape[-1]

    @property
    def num_layers(self) -> int:
        return self.X.shape[1] if self.X.ndim == 3 else 1

    def get_slice(
        self,
        anchor: Optional[str] = "post_gen",
        round_idx: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts a 2D feature matrix X [N_filtered, hidden_dim], labels y, and task groups.
        Args:
            anchor: Filter to a specific anchor name (e.g. 'post_gen' or 'pre_gen'). None for all.
            round_idx: Filter to a specific turn index t. None for all rounds.
            layer_idx: Slices a specific layer index from X. None defaults to first layer.
        Returns:
            X_2d: Array of shape [N_filtered, hidden_dim] in float32
            y_filtered: Array of shape [N_filtered] (int)
            groups: Array of shape [N_filtered] (task_idx)
        """
        mask = np.ones(self.num_samples, dtype=bool)

        if anchor is not None:
            mask &= (self.anchors == anchor)
        if round_idx is not None:
            mask &= (self.rounds == round_idx)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No samples matched filter (anchor={anchor}, round={round_idx})")

        if self.X.ndim == 3:
            l_pos = 0
            if layer_idx is not None:
                if layer_idx in self.layer_ids:
                    l_pos = self.layer_ids.index(layer_idx)
                else:
                    l_pos = min(max(0, layer_idx), self.X.shape[1] - 1)
            X_2d = self.X[indices, l_pos, :].astype(np.float32)
        else:
            X_2d = self.X[indices, :].astype(np.float32)

        return X_2d, self.y[indices], self.task_idx[indices]


def load_shards(feature_dir: str | Path) -> ShardDataset:
    """
    Loads all 'feat.shard*.npz' and matching 'meta.shard*.jsonl' files from a directory.
    """
    f_dir = Path(feature_dir)
    shard_files = sorted(glob.glob(str(f_dir / "feat.shard*.npz")))
    if not shard_files:
        # Check subfolder 'features' if passed parent run directory
        sub_shards = sorted(glob.glob(str(f_dir / "features" / "feat.shard*.npz")))
        if sub_shards:
            shard_files = sub_shards
        else:
            raise FileNotFoundError(f"No 'feat.shard*.npz' files found in {f_dir}")

    Xs: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    layer_ids: Optional[list[int]] = None

    for feat_file in shard_files:
        d = np.load(feat_file)
        Xs.append(d["X"])
        shard_layers = d["layer_ids"].tolist()
        if layer_ids is None:
            layer_ids = shard_layers
        else:
            assert layer_ids == shard_layers, "Mismatch in layer_ids across shards"

        meta_file = feat_file.replace("feat.shard", "meta.shard").replace(".npz", ".jsonl")
        with open(meta_file, "r", encoding="utf-8") as fp:
            metas.extend(json.loads(line) for line in fp if line.strip())

    X = np.concatenate(Xs, axis=0)
    assert X.shape[0] == len(metas), f"Tensor rows ({X.shape[0]}) != metadata count ({len(metas)})"

    y = np.array([int(m.get("success", 0)) for m in metas], dtype=np.int32)
    task_idx = np.array([int(m.get("task_idx", 0)) for m in metas], dtype=np.int32)
    rollout_k = np.array([int(m.get("rollout_k", 0)) for m in metas], dtype=np.int32)
    rounds = np.array([int(m.get("round", m.get("step_idx", 0))) for m in metas], dtype=np.int32)
    anchors = np.array([str(m.get("anchor", "post_gen")) for m in metas])

    return ShardDataset(
        X=X,
        y=y,
        task_idx=task_idx,
        rollout_k=rollout_k,
        rounds=rounds,
        anchors=anchors,
        layer_ids=layer_ids or [0],
        metas=metas,
    )


def extract_episodes(
    dataset: ShardDataset,
    target_layers: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    Parses a ShardDataset into a list of structured episode objects:
      ep = {
          "ep_id": str,
          "task_idx": int,
          "rollout_k": int,
          "success": int,
          "rounds": {
              round_idx: {
                  anchor_name: { layer_id: np.ndarray }
              }
          }
      }
    """
    if target_layers is not None:
        layer_dict = {l: dataset.layer_ids.index(l) for l in target_layers if l in dataset.layer_ids}
        if not layer_dict:
            layer_dict = {dataset.layer_ids[0]: 0}
    else:
        layer_dict = {l: i for i, l in enumerate(dataset.layer_ids)}

    episodes: dict[str, dict[str, Any]] = {}
    for i, m in enumerate(dataset.metas):
        ep_id = m.get("episode_id", f"{m.get('task_idx', 0)}_{m.get('rollout_k', 0)}")
        r = int(m.get("round", m.get("step_idx", 0)))
        anc = str(m.get("anchor", "post_gen"))

        if ep_id not in episodes:
            episodes[ep_id] = {
                "ep_id": ep_id,
                "task_idx": int(m.get("task_idx", 0)),
                "rollout_k": int(m.get("rollout_k", 0)),
                "success": int(m.get("success", 0)),
                "rounds": {},
            }

        if r not in episodes[ep_id]["rounds"]:
            episodes[ep_id]["rounds"][r] = {}

        episodes[ep_id]["rounds"][r][anc] = {
            layer_id: dataset.X[i, l_idx, :].astype(np.float32)
            for layer_id, l_idx in layer_dict.items()
        }

    return list(episodes.values())
