from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Replay:
    state: np.ndarray
    action_feature: np.ndarray
    action_index: np.ndarray
    reward: np.ndarray
    continued: np.ndarray
    next_state: np.ndarray
    user_id: np.ndarray
    transition_id: np.ndarray

    def __len__(self) -> int:
        return len(self.reward)

    def take(self, indices: np.ndarray) -> "Replay":
        return Replay(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "Replay":
        with np.load(path) as data:
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})


def stable_user_split(user_ids: np.ndarray, seed: int, train_fraction: float, validation_fraction: float) -> dict[str, np.ndarray]:
    users = np.unique(np.asarray(user_ids, dtype=np.int64))
    scores = np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}:{int(user)}".encode()).digest()[:8], "little")
        / float(2**64)
        for user in users
    ])
    return {
        "train": users[scores < train_fraction],
        "validation": users[(scores >= train_fraction) & (scores < train_fraction + validation_fraction)],
        "test": users[scores >= train_fraction + validation_fraction],
    }


def partition_training_replay(
    replay: Replay,
    seed: int,
    profile_train_fraction: float,
    reference_fraction: float,
) -> dict[str, Replay]:
    if profile_train_fraction + reference_fraction >= 1.0:
        raise ValueError("fractions must leave a nonempty audit partition")
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(replay))
    first = int(len(order) * profile_train_fraction)
    second = first + int(len(order) * reference_fraction)
    return {
        "base": replay.take(order[:first]),
        "reference": replay.take(order[first:second]),
        "audit": replay.take(order[second:]),
    }


def tensor_batch(replay: Replay, indices: np.ndarray, device: torch.device | str) -> dict[str, torch.Tensor]:
    selected = replay.take(np.asarray(indices))
    return {
        "state": torch.as_tensor(selected.state, dtype=torch.float32, device=device),
        "action_feature": torch.as_tensor(selected.action_feature, dtype=torch.float32, device=device),
        "action_index": torch.as_tensor(selected.action_index, dtype=torch.long, device=device),
        "reward": torch.as_tensor(selected.reward, dtype=torch.float32, device=device),
        "continued": torch.as_tensor(selected.continued, dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(selected.next_state, dtype=torch.float32, device=device),
        "user_id": torch.as_tensor(selected.user_id, dtype=torch.long, device=device),
        "transition_id": torch.as_tensor(selected.transition_id, dtype=torch.long, device=device),
    }


def assert_no_identity_overlap(*replays: Replay) -> None:
    seen: set[int] = set()
    for replay in replays:
        values = set(map(int, replay.transition_id))
        if seen.intersection(values):
            raise AssertionError("transition identity leakage across partitions")
        seen.update(values)
