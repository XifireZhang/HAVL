from __future__ import annotations

import unittest

import numpy as np

from bellman_sharing.data import Replay, assert_no_identity_overlap, partition_training_replay, stable_user_split


def replay(length: int) -> Replay:
    return Replay(
        state=np.zeros((length, 2), np.float32), action_feature=np.zeros((length, 2), np.float32),
        action_index=np.arange(length), reward=np.zeros(length), continued=np.ones(length),
        next_state=np.zeros((length, 2), np.float32), user_id=np.arange(length) % 3,
        transition_id=np.arange(length),
    )


class DataTests(unittest.TestCase):
    def test_user_split_is_disjoint_and_deterministic(self) -> None:
        users = np.arange(1, 1001)
        first = stable_user_split(users, 9, .8, .1)
        second = stable_user_split(users[::-1], 9, .8, .1)
        for key in first:
            np.testing.assert_array_equal(np.sort(first[key]), np.sort(second[key]))
        self.assertEqual(sum(map(len, first.values())), len(users))
        self.assertFalse(set(first["train"]) & set(first["test"]))

    def test_transition_partitions_have_no_leakage(self) -> None:
        parts = partition_training_replay(replay(100), 7, .7, .15)
        assert_no_identity_overlap(*parts.values())
        self.assertEqual(sum(map(len, parts.values())), 100)

    def test_leakage_check_rejects_overlap(self) -> None:
        value = replay(4)
        with self.assertRaises(AssertionError):
            assert_no_identity_overlap(value, value)


if __name__ == "__main__":
    unittest.main()
