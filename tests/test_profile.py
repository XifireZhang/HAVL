from __future__ import annotations

import unittest

import torch

from bellman_sharing.models import BellmanProfile
from bellman_sharing.profile import (
    SoftGrouping,
    bellman_profile_distance,
    profile_targets,
)


class ProfileTests(unittest.TestCase):
    def test_successor_target_respects_continuation_and_gamma(self) -> None:
        reward = torch.tensor([1.0, 0.0])
        continued = torch.tensor([1.0, 0.0])
        next_state = torch.ones(2, 3)
        projection = torch.tensor([[1.0], [0.0], [0.0]])
        target = profile_targets(reward, continued, next_state, projection, .9)
        torch.testing.assert_close(target["successor"].flatten(), torch.tensor([.9, 0.0]))
        self.assertFalse(target["successor"].requires_grad)

    def test_profile_distance_is_metric(self) -> None:
        a = torch.tensor([[0.0, .2, 1.0, 0.0]])
        b = torch.tensor([[1.0, .4, 0.0, 1.0]])
        c = torch.tensor([[2.0, .8, -1.0, 1.0]])
        args = (2.0, 3.0, .5, .9)
        dab = bellman_profile_distance(a, b, *args)
        dba = bellman_profile_distance(b, a, *args)
        dac = bellman_profile_distance(a, c, *args)
        dbc = bellman_profile_distance(b, c, *args)
        self.assertGreaterEqual(float(dab), 0.0)
        torch.testing.assert_close(dab, dba)
        self.assertLessEqual(float(dac), float(dab + dbc) + 1e-6)
        self.assertEqual(float(bellman_profile_distance(a, a, *args)), 0.0)

    def test_equal_current_value_can_hide_successor_difference(self) -> None:
        first = torch.tensor([[.5, .8, 1.0, 0.0]])
        second = torch.tensor([[.5, .8, -1.0, 0.0]])
        distance = bellman_profile_distance(first, second, 1.0, 1.0, 1.0, .9)
        self.assertGreater(float(distance), 0.0)

    def test_soft_grouping_detaches_and_marks_unsupported(self) -> None:
        grouping = SoftGrouping(
            prototypes=torch.tensor([[0.0, .5, 0.0], [1.0, .5, 0.0]]),
            radius=.2, temperature=.1, scales=(1.0, 1.0, 1.0), gamma=.9, ema=.1,
        )
        vectors = torch.tensor([[0.05, .5, 0.0], [5.0, .5, 0.0]], requires_grad=True)
        weights, unsupported = grouping.weights(vectors)
        self.assertFalse(weights.requires_grad)
        self.assertFalse(bool(unsupported[0]))
        self.assertTrue(bool(unsupported[1]))
        torch.testing.assert_close(weights[1], torch.zeros(2))

    def test_profile_heads_have_expected_shapes(self) -> None:
        model = BellmanProfile(5, 4, 3, 8)
        output = model(torch.randn(7, 5), torch.randn(7, 4))
        self.assertEqual(output["reward"].shape, (7,))
        self.assertEqual(output["continue_logit"].shape, (7,))
        self.assertEqual(output["successor"].shape, (7, 3))


if __name__ == "__main__":
    unittest.main()
