from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class QCritic(nn.Module):
    """Nonlinear state-action critic; all layers participate in each update."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action_feature: torch.Tensor) -> torch.Tensor:
        if state.shape[:-1] != action_feature.shape[:-1]:
            state = state.expand(*action_feature.shape[:-1], state.shape[-1])
        return self.network(torch.cat([state, action_feature], dim=-1)).squeeze(-1)


class PositiveQError(nn.Module):
    """Positive state-action error estimator used by the DisCor adapter."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action_feature: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(
            self.network(torch.cat([state, action_feature], dim=-1)).squeeze(-1)
        )


class ValueCritic(nn.Module):
    """State-value critic used by A2C; all parameters are controller-visible."""

    def __init__(self, state_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = mlp(state_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class CatalogActor(nn.Module):
    """A2C actor that scores the complete catalog with a learned state query."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        initial_logit_scale: float = 10.0,
    ):
        super().__init__()
        self.query = mlp(state_dim, hidden_dims, action_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale)))

    def forward(
        self, state: torch.Tensor, normalized_catalog_features: torch.Tensor
    ) -> torch.Tensor:
        query = torch.nn.functional.normalize(self.query(state), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * query @ normalized_catalog_features.T


class BellmanProfile(nn.Module):
    """Predict r, continuation probability, and joint discounted successor feature."""

    def __init__(self, state_dim: int, action_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = mlp(state_dim + action_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.continue_head = nn.Linear(hidden_dim, 1)
        self.successor_head = nn.Linear(hidden_dim, feature_dim)

    def forward(self, state: torch.Tensor, action_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([state, action_feature], dim=-1))
        return {
            "reward": self.reward_head(hidden).squeeze(-1),
            "continue_logit": self.continue_head(hidden).squeeze(-1),
            "successor": self.successor_head(hidden),
        }


@torch.no_grad()
def max_sampled_q(
    critic: QCritic,
    states: torch.Tensor,
    candidate_features: torch.Tensor,
) -> torch.Tensor:
    batch, candidates, _ = candidate_features.shape
    expanded_state = states[:, None, :].expand(batch, candidates, states.shape[-1])
    return critic(expanded_state, candidate_features).max(dim=1).values


@torch.no_grad()
def greedy_full_catalog(
    critic: QCritic,
    states: torch.Tensor,
    catalog_features: torch.Tensor,
    chunk_size: int,
    forbidden: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    best_value = torch.full((len(states),), -torch.inf, device=states.device)
    best_index = torch.zeros(len(states), dtype=torch.long, device=states.device)
    for start in range(0, len(catalog_features), chunk_size):
        end = min(start + chunk_size, len(catalog_features))
        actions = catalog_features[start:end]
        values = critic(
            states[:, None, :].expand(len(states), len(actions), states.shape[-1]),
            actions[None, :, :].expand(len(states), len(actions), actions.shape[-1]),
        )
        if forbidden is not None:
            for row, blocked in enumerate(forbidden):
                local = blocked[(blocked >= start) & (blocked < end)] - start
                values[row, local] = -torch.inf
        chunk_value, chunk_index = values.max(dim=1)
        replace = chunk_value > best_value
        best_value[replace] = chunk_value[replace]
        best_index[replace] = start + chunk_index[replace]
    return best_index
