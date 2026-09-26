from __future__ import annotations

import numpy as np
import torch
import unittest

from bellman_sharing.main_table import (
    OnlineReplay,
    apply_controlled_critic_step,
    effective_sample_size,
    equal_user_mean,
    sampled_max_target,
)
from bellman_sharing.models import QCritic


class MainTableTests(unittest.TestCase):
    def test_effective_sample_size_distinguishes_supported_groups(self) -> None:
        weights = torch.tensor([
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        ess = effective_sample_size(weights)
        torch.testing.assert_close(ess, torch.tensor([4.0, 1.0]))

    def test_equal_user_metric_does_not_overweight_active_user(self) -> None:
        # User 1 supplies three sessions; user 2 supplies one. Each user has weight 1/2.
        self.assertEqual(equal_user_mean([1.0, 1.0, 1.0, 5.0], [1, 1, 1, 2]), 3.0)

    def test_terminal_transition_does_not_bootstrap(self) -> None:
        critic = QCritic(2, 2, [4])
        for parameter in critic.parameters():
            torch.nn.init.constant_(parameter, 0.5)
        batch = {
            "state": torch.zeros(2, 2),
            "action_feature": torch.zeros(2, 2),
            "reward": torch.tensor([2.0, 2.0]),
            "continued": torch.tensor([0.0, 1.0]),
            "next_state": torch.ones(2, 2),
        }
        catalog = torch.ones(3, 2)
        target = sampled_max_target(critic, batch, catalog, torch.arange(3), 0.9)
        self.assertEqual(target[0].item(), 2.0)
        self.assertGreater(target[1].item(), 2.0)

    def test_online_replay_user_uniform_sampling_keeps_shapes(self) -> None:
        replay = OnlineReplay(8, 2, 3, torch.device("cpu"))
        replay.append({
            "state": torch.zeros(4, 2),
            "action_feature": torch.zeros(4, 3),
            "action_index": torch.arange(4),
            "reward": torch.arange(4, dtype=torch.float32),
            "continued": torch.ones(4),
            "next_state": torch.ones(4, 2),
            "user_id": torch.tensor([1, 1, 1, 2]),
            "transition_id": torch.arange(4),
        })
        batch = replay.sample_equal_user(6, np.random.default_rng(7))
        self.assertEqual(batch["state"].shape, (6, 2))
        self.assertTrue(set(batch["user_id"].tolist()).issubset({1, 2}))

    def test_zero_controlled_step_restores_adam_state(self) -> None:
        critic = QCritic(1, 1, [])
        for parameter in critic.parameters():
            torch.nn.init.zeros_(parameter)
        optimizer = torch.optim.Adam(critic.parameters(), lr=0.1)
        batch = {
            "state": torch.zeros(4, 1),
            "action_feature": torch.zeros(4, 1),
        }
        before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
        loss = torch.nn.functional.mse_loss(
            critic(batch["state"], batch["action_feature"]), torch.ones(4)
        )
        result = apply_controlled_critic_step(
            critic, optimizer, loss, batch, -torch.ones(4), torch.ones(4, 1),
            {
                "min_group_ess": 2.0,
                "budget_eta": 0.0,
                "loss_floor": 0.1,
                "trust_radius_ratio": 1.0,
                "numerical_tolerance": 1e-10,
                "backtracking_steps": 2,
            },
        )
        self.assertTrue(result["zero_update"])
        self.assertEqual(len(optimizer.state), 0)
        for name, value in critic.state_dict().items():
            torch.testing.assert_close(value, before[name])


if __name__ == "__main__":
    unittest.main()
