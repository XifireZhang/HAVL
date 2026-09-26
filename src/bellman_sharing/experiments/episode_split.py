from __future__ import annotations

import numpy as np

from ..data import Replay


def episode_keys(replay: Replay, batch_size: int) -> np.ndarray:
    """Reconstruct per-stream episodes from ordered transition ids and done flags."""
    order = np.argsort(replay.transition_id)
    keys = np.empty(len(replay), dtype=np.int64)
    episode_count = np.zeros(batch_size, dtype=np.int64)
    for index in order:
        stream = int(replay.transition_id[index] % batch_size)
        keys[index] = stream + batch_size * episode_count[stream]
        if not bool(replay.continued[index]):
            episode_count[stream] += 1
    return keys


def partition_replay_by_episode(
    replay: Replay,
    batch_size: int,
    seed: int,
    training_fraction: float,
    reference_fraction: float,
) -> tuple[dict[str, Replay], dict[str, np.ndarray]]:
    if training_fraction + reference_fraction >= 1.0:
        raise ValueError("fractions must leave a nonempty audit partition")
    keys = episode_keys(replay, batch_size)
    episodes = np.unique(keys)
    order = np.random.default_rng(seed).permutation(episodes)
    first = int(len(order) * training_fraction)
    second = first + int(len(order) * reference_fraction)
    role_keys = {
        "training": order[:first],
        "reference": order[first:second],
        "audit": order[second:],
    }
    partitions = {
        name: replay.take(np.flatnonzero(np.isin(keys, selected)))
        for name, selected in role_keys.items()
    }
    return partitions, role_keys

