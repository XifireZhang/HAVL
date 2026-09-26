from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .models import BellmanProfile


def orthogonal_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("output_dim cannot exceed input_dim")
    generator = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(generator.normal(size=(input_dim, output_dim)))
    return torch.as_tensor(basis[:, :output_dim], dtype=torch.float32)


def profile_targets(
    reward: torch.Tensor,
    continued: torch.Tensor,
    next_state: torch.Tensor,
    projection: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    feature = next_state @ projection.to(next_state.device)
    return {
        "reward": reward.detach(),
        "continued": continued.detach(),
        "successor": (gamma * continued[:, None] * feature).detach(),
    }


def profile_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    weights: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    reward = F.mse_loss(prediction["reward"], target["reward"])
    continuation = F.binary_cross_entropy_with_logits(
        prediction["continue_logit"], target["continued"]
    )
    successor = F.mse_loss(prediction["successor"], target["successor"])
    total = weights[0] * reward + weights[1] * continuation + weights[2] * successor
    return total, {
        "reward_mse": float(reward.detach()),
        "continue_bce": float(continuation.detach()),
        "successor_mse": float(successor.detach()),
    }


@torch.no_grad()
def profile_vector(model: BellmanProfile, state: torch.Tensor, action_feature: torch.Tensor) -> torch.Tensor:
    output = model(state, action_feature)
    return torch.cat([
        output["reward"][:, None],
        torch.sigmoid(output["continue_logit"])[:, None],
        output["successor"],
    ], dim=1)


def bellman_profile_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    reward_scale: float,
    continue_scale: float,
    successor_scale: float,
    gamma: float,
) -> torch.Tensor:
    reward = reward_scale * (left[..., 0] - right[..., 0]).abs()
    continuation = continue_scale * gamma * (left[..., 1] - right[..., 1]).abs()
    successor = successor_scale * torch.linalg.vector_norm(left[..., 2:] - right[..., 2:], dim=-1)
    return reward + continuation + successor


def farthest_point_prototypes(vectors: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or count > len(vectors):
        raise ValueError("invalid prototype count")
    selected = [0]
    nearest = torch.linalg.vector_norm(vectors - vectors[0], dim=1)
    for _ in range(1, count):
        index = int(torch.argmax(nearest))
        selected.append(index)
        nearest = torch.minimum(nearest, torch.linalg.vector_norm(vectors - vectors[index], dim=1))
    return vectors[selected].clone()


@dataclass
class SoftGrouping:
    prototypes: torch.Tensor
    radius: float
    temperature: float
    scales: tuple[float, float, float]
    gamma: float
    ema: float

    def weights(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distance = bellman_profile_distance(
            vectors[:, None, :], self.prototypes[None, :, :],
            *self.scales, self.gamma,
        )
        supported = distance <= self.radius
        logits = -distance / max(self.temperature, 1e-8)
        logits = logits.masked_fill(~supported, -torch.inf)
        ungrouped = ~supported.any(dim=1)
        logits[ungrouped] = 0.0
        weights = torch.softmax(logits, dim=1)
        weights[ungrouped] = 0.0
        return weights.detach(), ungrouped.detach()

    @torch.no_grad()
    def slow_update(self, vectors: torch.Tensor, weights: torch.Tensor) -> None:
        mass = weights.sum(dim=0)
        for group in range(len(self.prototypes)):
            if mass[group] > 0:
                center = (weights[:, group, None] * vectors).sum(dim=0) / mass[group]
                self.prototypes[group].lerp_(center, self.ema)


def fit_grouping(
    vectors: torch.Tensor,
    validation_vectors: torch.Tensor,
    count: int,
    radius_quantile: float,
    temperature: float,
    gamma: float,
    ema: float,
) -> SoftGrouping:
    scale = vectors.std(dim=0).clamp_min(1e-4)
    reward_scale = float(1.0 / scale[0])
    continue_scale = float(1.0 / scale[1])
    successor_scale = float(1.0 / scale[2:].square().mean().sqrt())
    normalized = vectors / scale
    prototypes = farthest_point_prototypes(normalized, count) * scale
    distances = bellman_profile_distance(
        validation_vectors[:, None, :], prototypes[None, :, :],
        reward_scale, continue_scale, successor_scale, gamma,
    )
    radius = float(torch.quantile(distances.min(dim=1).values, radius_quantile))
    return SoftGrouping(
        prototypes=prototypes.detach(), radius=radius, temperature=temperature,
        scales=(reward_scale, continue_scale, successor_scale), gamma=gamma, ema=ema,
    )
