from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .controller import (
    actual_optimizer_candidate,
    apply_with_loss_budgets,
    assign_flat,
    flat_gradient,
    project_candidate,
    trainable_parameters,
)
from .data import Replay
from .models import BellmanProfile, PositiveQError, QCritic, greedy_full_catalog, max_sampled_q
from .profile import (
    SoftGrouping,
    farthest_point_prototypes,
    fit_grouping,
    profile_loss,
    profile_targets,
    profile_vector,
)


class OnlineReplay:
    """A fixed-capacity GPU replay with ordinary and equal-user sampling."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.state = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.action_feature = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.action_index = torch.empty(capacity, dtype=torch.long, device=device)
        self.reward = torch.empty(capacity, dtype=torch.float32, device=device)
        self.continued = torch.empty(capacity, dtype=torch.float32, device=device)
        self.next_state = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.user_id = torch.empty(capacity, dtype=torch.long, device=device)
        self.transition_id = torch.empty(capacity, dtype=torch.long, device=device)
        self.size = 0
        self.user_indices: dict[int, list[int]] = defaultdict(list)

    def __len__(self) -> int:
        return self.size

    def append(self, values: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(len(values["reward"]), dtype=torch.bool, device=self.device)
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(len(selected))
        if self.size + count > self.capacity:
            raise RuntimeError(f"replay capacity exceeded: {self.size + count} > {self.capacity}")
        destination = slice(self.size, self.size + count)
        for name in (
            "state", "action_feature", "action_index", "reward", "continued",
            "next_state", "user_id", "transition_id",
        ):
            getattr(self, name)[destination].copy_(values[name][selected].detach())
        users = values["user_id"][selected].detach().cpu().tolist()
        for offset, user in enumerate(users):
            self.user_indices[int(user)].append(self.size + offset)
        self.size += count

    def append_replay(self, replay: Replay) -> None:
        values = {
            "state": torch.as_tensor(replay.state, device=self.device),
            "action_feature": torch.as_tensor(replay.action_feature, device=self.device),
            "action_index": torch.as_tensor(replay.action_index, device=self.device),
            "reward": torch.as_tensor(replay.reward, device=self.device),
            "continued": torch.as_tensor(replay.continued, device=self.device),
            "next_state": torch.as_tensor(replay.next_state, device=self.device),
            "user_id": torch.as_tensor(replay.user_id, device=self.device),
            "transition_id": torch.as_tensor(replay.transition_id, device=self.device),
        }
        self.append(values)

    def batch(self, indices: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return {
            name: getattr(self, name)[index]
            for name in (
                "state", "action_feature", "action_index", "reward", "continued",
                "next_state", "user_id", "transition_id",
            )
        }

    def sample(self, count: int, generator: np.random.Generator) -> dict[str, torch.Tensor]:
        return self.batch(generator.integers(0, self.size, size=int(count), dtype=np.int64))

    def sample_equal_user(
        self, count: int, generator: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        users = np.asarray(list(self.user_indices), dtype=np.int64)
        if not len(users):
            raise RuntimeError("cannot sample an empty replay")
        chosen_users = generator.choice(users, size=int(count), replace=len(users) < count)
        indices = [
            self.user_indices[int(user)][generator.integers(0, len(self.user_indices[int(user)]))]
            for user in chosen_users
        ]
        return self.batch(np.asarray(indices, dtype=np.int64))


@dataclass
class EuclideanGrouping:
    prototypes: torch.Tensor
    scale: torch.Tensor
    radius: float
    temperature: float
    ema: float

    def weights(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = vectors / self.scale
        centers = self.prototypes / self.scale
        distance = torch.cdist(normalized, centers)
        supported = distance <= self.radius
        logits = (-distance / max(self.temperature, 1e-8)).masked_fill(~supported, -torch.inf)
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


def fit_euclidean_grouping(
    vectors: torch.Tensor,
    calibration: torch.Tensor,
    count: int,
    radius_quantile: float,
    temperature: float,
    ema: float,
) -> EuclideanGrouping:
    scale = vectors.std(dim=0).clamp_min(1e-4)
    prototypes = farthest_point_prototypes(vectors / scale, count) * scale
    distances = torch.cdist(calibration / scale, prototypes / scale)
    radius = float(torch.quantile(distances.min(dim=1).values, radius_quantile))
    return EuclideanGrouping(prototypes.detach(), scale.detach(), radius, temperature, ema)


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    mass = weights.sum(dim=0)
    return mass.square() / weights.square().sum(dim=0).clamp_min(1e-12)


def weighted_td_losses(
    critic: QCritic,
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
    weights: torch.Tensor,
    active_groups: torch.Tensor,
) -> torch.Tensor:
    per_sample = 0.5 * (critic(batch["state"], batch["action_feature"]) - target).square()
    return torch.stack([
        (weights[:, group] * per_sample).sum() / weights[:, group].sum().clamp_min(1e-12)
        for group in active_groups
    ]) if len(active_groups) else torch.empty(0, device=per_sample.device)


def sampled_max_target(
    target: QCritic,
    batch: dict[str, torch.Tensor],
    catalog: torch.Tensor,
    candidate_indices: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    candidate = catalog[candidate_indices][None, :, :].expand(len(batch["state"]), -1, -1)
    return batch["reward"] + gamma * batch["continued"] * max_sampled_q(
        target, batch["next_state"], candidate
    )


@torch.no_grad()
def sampled_target_with_greedy_feature(
    target: QCritic,
    batch: dict[str, torch.Tensor],
    catalog: torch.Tensor,
    candidate_indices: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = catalog[candidate_indices]
    candidate = features[None, :, :].expand(len(batch["state"]), -1, -1)
    values = target(
        batch["next_state"][:, None, :].expand(
            len(batch["state"]), len(features), batch["state"].shape[-1]
        ),
        candidate,
    )
    best_value, best_index = values.max(dim=1)
    y = batch["reward"] + gamma * batch["continued"] * best_value
    return y, features[best_index]


@torch.no_grad()
def sampled_epsilon_greedy(
    critic: QCritic,
    state: torch.Tensor,
    observation: dict,
    adapter: Any,
    candidate_count: int,
    epsilon: float,
    generator: torch.Generator,
) -> torch.Tensor:
    candidate = torch.randint(
        len(adapter.catalog_features), (int(candidate_count),),
        generator=generator, device=state.device,
    )
    features = adapter.catalog_features[candidate]
    values = critic(
        state[:, None, :].expand(len(state), len(candidate), state.shape[-1]),
        features[None, :, :].expand(len(state), len(candidate), features.shape[-1]),
    )
    history = observation["user_history"]["history"]
    encoded = adapter.env.candidate_iids[candidate]
    collision = (history[:, :, None] == encoded[None, None, :]).any(dim=1)
    values.masked_fill_(collision, -torch.inf)
    greedy = candidate[values.argmax(dim=1)]
    if epsilon <= 0:
        return greedy
    explore = torch.rand(len(state), generator=generator, device=state.device) < epsilon
    if explore.any():
        random_action = adapter.random_actions(observation, generator)
        greedy[explore] = random_action[explore]
    return greedy


def transition_values(
    adapter: Any,
    observation: dict,
    action: torch.Tensor,
    next_observation: dict,
    response: dict,
    transition_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "state": adapter.state(observation),
        "action_feature": adapter.catalog_features[action],
        "action_index": action,
        "reward": response["immediate_response"][:, 0, 0],
        "continued": (~response["done"]).float(),
        "next_state": adapter.state(next_observation),
        "user_id": observation["user_profile"]["user_id"].long(),
        "transition_id": transition_ids.long(),
    }


def random_descriptor(transition_id: torch.Tensor, seed: int, width: int = 4) -> torch.Tensor:
    value = transition_id.float()[:, None]
    frequencies = torch.arange(1, width + 1, device=value.device, dtype=value.dtype)[None]
    return torch.sin(value * (0.0137 * frequencies) + float(seed) * 0.001)


def descriptor_for_method(
    method: str,
    batch: dict[str, torch.Tensor],
    teacher: BellmanProfile,
    target: QCritic,
    user_counts: dict[int, int],
    random_seed: int,
) -> torch.Tensor:
    with torch.no_grad():
        if method == "havl":
            return profile_vector(teacher, batch["state"], batch["action_feature"])
        if method == "havl_value":
            return target(batch["state"], batch["action_feature"])[:, None]
        if method == "havl_activity":
            counts = [math.log1p(user_counts.get(int(user), 0)) for user in batch["user_id"].cpu()]
            return torch.tensor(counts, dtype=torch.float32, device=batch["state"].device)[:, None]
        if method == "havl_random":
            return random_descriptor(batch["transition_id"], random_seed)
    raise ValueError(method)


def build_grouping(
    method: str,
    reference: dict[str, torch.Tensor],
    calibration: dict[str, torch.Tensor],
    teacher: BellmanProfile,
    target: QCritic,
    user_counts: dict[int, int],
    config: dict[str, Any],
    seed: int,
) -> SoftGrouping | EuclideanGrouping:
    vectors = descriptor_for_method(method, reference, teacher, target, user_counts, seed)
    validation = descriptor_for_method(method, calibration, teacher, target, user_counts, seed)
    if method == "havl":
        return fit_grouping(
            vectors, validation, int(config["groups"]), float(config["radius_quantile"]),
            float(config["soft_temperature"]), float(config["gamma"]),
            float(config["prototype_ema"]),
        )
    return fit_euclidean_grouping(
        vectors, validation, int(config["groups"]), float(config["radius_quantile"]),
        float(config["soft_temperature"]), float(config["prototype_ema"]),
    )


def soft_update(source: torch.nn.Module, destination: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for parameter, target_parameter in zip(source.parameters(), destination.parameters()):
            target_parameter.lerp_(parameter, float(tau))


def update_profile(
    model: BellmanProfile,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    projection: torch.Tensor,
    config: dict[str, Any],
) -> float:
    target = profile_targets(
        batch["reward"], batch["continued"], batch["next_state"], projection,
        float(config["gamma"]),
    )
    prediction = model(batch["state"], batch["action_feature"])
    loss, _ = profile_loss(
        prediction, target,
        (float(config["weight_reward"]), float(config["weight_continue"]),
         float(config["weight_successor"])),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def apply_controlled_critic_step(
    critic: QCritic,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    reference: dict[str, torch.Tensor],
    reference_target: torch.Tensor,
    weights: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    optimizer_before = deepcopy(optimizer.state_dict())
    original, natural = actual_optimizer_candidate(critic, optimizer, loss)
    natural_norm = float(torch.linalg.vector_norm(natural))
    ess = effective_sample_size(weights)
    active = torch.nonzero(
        ess >= float(config["min_group_ess"]), as_tuple=False
    ).flatten()
    if not len(active):
        assign_flat(trainable_parameters(critic), original + natural)
        return {
            "controlled": True, "active_groups": 0, "natural_norm": natural_norm,
            "norm_ratio": 1.0, "cosine": 1.0, "alpha": 1.0,
            "zero_update": False, "protected_fraction": 0.0,
            "min_ess": float(ess.min()) if len(ess) else 0.0,
        }
    losses = weighted_td_losses(critic, reference, reference_target, weights, active)
    gradients = torch.stack([
        flat_gradient(group_loss, critic, retain_graph=slot + 1 < len(losses))
        for slot, group_loss in enumerate(losses)
    ])
    tolerance = float(config["budget_eta"]) * losses.detach().clamp_min(
        float(config["loss_floor"])
    )
    projection = project_candidate(
        natural,
        gradients,
        tolerance / 2.0,
        float(config["trust_radius_ratio"]) * natural_norm,
    )

    def closure() -> torch.Tensor:
        return weighted_td_losses(critic, reference, reference_target, weights, active)

    backtracking = apply_with_loss_budgets(
        critic, original, projection.displacement, closure, tolerance,
        float(config["numerical_tolerance"]), int(config["backtracking_steps"]),
    )
    applied = backtracking.displacement
    applied_norm = float(torch.linalg.vector_norm(applied))
    if not backtracking.accepted or applied_norm == 0.0:
        optimizer.load_state_dict(optimizer_before)
    cosine = float(torch.dot(applied, natural) / max(applied_norm * natural_norm, 1e-24))
    protected_mass = weights[:, active].sum(dim=1).clamp(max=1.0)
    return {
        "controlled": True,
        "active_groups": int(len(active)),
        "natural_norm": natural_norm,
        "norm_ratio": applied_norm / max(natural_norm, 1e-24),
        "cosine": cosine,
        "alpha": float(backtracking.scale),
        "zero_update": (not backtracking.accepted) or applied_norm == 0.0,
        "protected_fraction": float((protected_mass > 0).float().mean()),
        "min_active_ess": float(ess[active].min()),
        "maximum_linear_violation": projection.maximum_linear_violation,
        "projection_converged": projection.converged,
        "maximum_actual_increase": float(
            (backtracking.losses_after - backtracking.losses_before).max()
        ),
        "maximum_budget": float(tolerance.max()),
    }


def apply_gem_critic_step(
    critic: QCritic,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    reference: dict[str, torch.Tensor],
    reference_target: torch.Tensor,
    weights: torch.Tensor,
    min_group_ess: float,
) -> dict[str, Any]:
    """Project an actual optimizer update onto GEM-style zero-increase halfspaces."""
    optimizer_before = deepcopy(optimizer.state_dict())
    original, natural = actual_optimizer_candidate(critic, optimizer, loss)
    natural_norm = float(torch.linalg.vector_norm(natural))
    ess = effective_sample_size(weights)
    active = torch.nonzero(ess >= float(min_group_ess), as_tuple=False).flatten()
    if not len(active):
        assign_flat(trainable_parameters(critic), original + natural)
        return {
            "controlled": True, "active_groups": 0, "natural_norm": natural_norm,
            "norm_ratio": 1.0, "cosine": 1.0, "alpha": 1.0,
            "zero_update": False, "protected_fraction": 0.0,
        }
    losses = weighted_td_losses(critic, reference, reference_target, weights, active)
    gradients = torch.stack([
        flat_gradient(group_loss, critic, retain_graph=slot + 1 < len(losses))
        for slot, group_loss in enumerate(losses)
    ])
    projection = project_candidate(
        natural, gradients, torch.zeros(len(active), device=natural.device), float("inf")
    )
    applied = projection.displacement
    applied_norm = float(torch.linalg.vector_norm(applied))
    assign_flat(trainable_parameters(critic), original + applied)
    if applied_norm == 0.0:
        optimizer.load_state_dict(optimizer_before)
    cosine = float(torch.dot(applied, natural) / max(applied_norm * natural_norm, 1e-24))
    protected_mass = weights[:, active].sum(dim=1).clamp(max=1.0)
    return {
        "controlled": True, "active_groups": int(len(active)),
        "natural_norm": natural_norm,
        "norm_ratio": applied_norm / max(natural_norm, 1e-24),
        "cosine": cosine, "alpha": 1.0, "zero_update": applied_norm == 0.0,
        "protected_fraction": float((protected_mass > 0).float().mean()),
        "min_active_ess": float(ess[active].min()),
        "maximum_linear_violation": projection.maximum_linear_violation,
        "projection_converged": projection.converged,
    }


def fixed_activity_weights(
    users: torch.Tensor,
    fixed_counts: dict[int, int],
    boundaries: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    values = torch.tensor(
        [math.log1p(fixed_counts.get(int(user), 0)) for user in users.cpu()],
        dtype=torch.float32, device=users.device,
    )
    assignment = torch.bucketize(values, boundaries)
    return F.one_hot(assignment, num_classes=int(groups)).float()


def train_online_dqn(
    method: str,
    adapter: Any,
    warmup_main: Replay,
    warmup_reference: Replay,
    initial_critic_state: dict[str, torch.Tensor],
    initial_profile_state: dict[str, torch.Tensor],
    projection: torch.Tensor,
    config: dict[str, Any],
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[QCritic, dict[str, Any]]:
    if method not in {
        "base", "user_uniform", "gem_adapted", "discor_adapted", "havl_random",
        "havl_activity", "havl_value", "havl"
    }:
        raise ValueError(method)
    device = adapter.device
    torch.manual_seed(seed)
    np_generator = np.random.default_rng(seed + 1)
    torch_generator = torch.Generator(device=device).manual_seed(seed + 2)
    critic = QCritic(adapter.state_dim, adapter.action_feature_dim, config["hidden_dims"]).to(device)
    critic.load_state_dict(initial_critic_state)
    target = deepcopy(critic).eval()
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    profile = BellmanProfile(
        adapter.state_dim, adapter.action_feature_dim, projection.shape[1],
        int(config["profile_hidden_dim"]),
    ).to(device)
    profile.load_state_dict(initial_profile_state)
    teacher = deepcopy(profile).eval()
    profile_optimizer = torch.optim.Adam(profile.parameters(), lr=float(config["profile_learning_rate"]))
    error_model = None
    error_target = None
    error_optimizer = None
    error_scale = 1.0
    if method == "discor_adapted":
        error_model = PositiveQError(
            adapter.state_dim, adapter.action_feature_dim, config["hidden_dims"]
        ).to(device)
        error_target = deepcopy(error_model).eval()
        error_optimizer = torch.optim.Adam(
            error_model.parameters(), lr=float(config.get("discor_learning_rate", 0.001))
        )
    capacity = int(config["replay_capacity"])
    replay = OnlineReplay(capacity, adapter.state_dim, adapter.action_feature_dim, device)
    reference_replay = OnlineReplay(capacity, adapter.state_dim, adapter.action_feature_dim, device)
    replay.append_replay(warmup_main)
    reference_replay.append_replay(warmup_reference)
    user_counts: dict[int, int] = defaultdict(int)
    for user in np.concatenate([warmup_main.user_id, warmup_reference.user_id]):
        user_counts[int(user)] += 1
    fixed_activity_counts = dict(user_counts)
    activity_values = torch.tensor(
        [math.log1p(value) for value in fixed_activity_counts.values()],
        dtype=torch.float32, device=device,
    )
    activity_boundaries = torch.quantile(
        activity_values, torch.linspace(0, 1, int(config["groups"]) + 1, device=device)
    )[1:-1]
    observation = adapter.reset("train", seed + 3)
    transition_counter = int(max(warmup_main.transition_id.max(), warmup_reference.transition_id.max())) + 1
    grouping = None
    records: list[dict[str, Any]] = []
    controlled_method = method.startswith("havl")
    updates = int(config["updates"])
    for update in range(updates):
        fraction = update / max(updates - 1, 1)
        epsilon = float(config["epsilon_start"]) + fraction * (
            float(config["epsilon_end"]) - float(config["epsilon_start"])
        )
        state = adapter.state(observation)
        action = sampled_epsilon_greedy(
            critic, state, observation, adapter, int(config["behavior_candidate_count"]),
            epsilon, torch_generator,
        )
        current_observation = observation
        next_observation, response, _ = adapter.env.step({"action": action[:, None]})
        identifiers = torch.arange(
            transition_counter, transition_counter + adapter.batch_size, device=device
        )
        transition_counter += adapter.batch_size
        values = transition_values(
            adapter, current_observation, action, next_observation, response, identifiers
        )
        reference_mask = torch.remainder(identifiers, int(config["reference_stride"])) == 0
        reference_replay.append(values, reference_mask)
        replay.append(values, ~reference_mask)
        for user in values["user_id"].detach().cpu().tolist():
            user_counts[int(user)] += 1
        observation = next_observation

        batch = (
            replay.sample_equal_user(int(config["batch_size"]), np_generator)
            if method == "user_uniform"
            else replay.sample(int(config["batch_size"]), np_generator)
        )
        candidate_indices = torch.randint(
            len(adapter.catalog_features), (int(config["target_candidate_count"]),),
            generator=torch_generator, device=device,
        )
        if method == "discor_adapted":
            assert error_model is not None and error_target is not None and error_optimizer is not None
            y, greedy_feature = sampled_target_with_greedy_feature(
                target, batch, adapter.catalog_features, candidate_indices, float(config["gamma"])
            )
            with torch.no_grad():
                next_error = error_target(batch["next_state"], greedy_feature)
                temperature = max(
                    float(config.get("discor_temperature", 1.0)) * error_scale, 1e-4
                )
                logits = -float(config["gamma"]) * next_error / temperature
                weights = torch.softmax(logits, dim=0) * len(logits)
            q = critic(batch["state"], batch["action_feature"])
            loss = (weights * (q - y).square()).mean()
            with torch.no_grad():
                error_label = (
                    (q.detach() - y).abs()
                    + float(config["gamma"]) * batch["continued"] * next_error
                )
            predicted_error = error_model(batch["state"], batch["action_feature"])
            error_loss = 0.5 * F.mse_loss(predicted_error, error_label)
            error_optimizer.zero_grad(set_to_none=True)
            error_loss.backward()
            error_optimizer.step()
            error_scale = (
                float(config.get("discor_scale_ema", 0.99)) * error_scale
                + (1.0 - float(config.get("discor_scale_ema", 0.99)))
                * float(predicted_error.detach().mean())
            )
        else:
            with torch.no_grad():
                y = sampled_max_target(
                    target, batch, adapter.catalog_features, candidate_indices,
                    float(config["gamma"]),
                )
            loss = F.mse_loss(critic(batch["state"], batch["action_feature"]), y)
        record: dict[str, Any] = {
            "update": update, "loss": float(loss.detach()), "epsilon": epsilon,
            "controlled": False, "norm_ratio": 1.0, "cosine": 1.0,
            "alpha": 1.0, "zero_update": False,
        }
        if method == "discor_adapted":
            record.update({
                "error_loss": float(error_loss.detach()),
                "error_scale": float(error_scale),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "weight_ess": float(weights.sum().square() / weights.square().sum()),
            })
        if controlled_method and update % int(config["control_interval"]) == 0:
            refresh = grouping is None or update % int(config["prototype_refresh_interval"]) == 0
            if refresh:
                fit_reference = reference_replay.sample_equal_user(
                    int(config["prototype_fit_size"]), np_generator
                )
                calibration = reference_replay.sample_equal_user(
                    int(config["prototype_calibration_size"]), np_generator
                )
                grouping = build_grouping(
                    method, fit_reference, calibration, teacher, target, user_counts,
                    config, seed + 11,
                )
            reference = reference_replay.sample_equal_user(
                int(config["reference_batch_size"]), np_generator
            )
            reference_candidates = torch.randint(
                len(adapter.catalog_features), (int(config["target_candidate_count"]),),
                generator=torch_generator, device=device,
            )
            with torch.no_grad():
                reference_target = sampled_max_target(
                    target, reference, adapter.catalog_features, reference_candidates,
                    float(config["gamma"]),
                )
                descriptor = descriptor_for_method(
                    method, reference, teacher, target, user_counts, seed + 11
                )
                weights, ungrouped = grouping.weights(descriptor)
            record.update(apply_controlled_critic_step(
                critic, optimizer, loss, reference, reference_target, weights, config
            ))
            record["ungrouped_fraction"] = float(ungrouped.float().mean())
            grouping.slow_update(descriptor, weights)
        elif method == "gem_adapted" and update % int(config["control_interval"]) == 0:
            reference = reference_replay.sample_equal_user(
                int(config["reference_batch_size"]), np_generator
            )
            reference_candidates = torch.randint(
                len(adapter.catalog_features), (int(config["target_candidate_count"]),),
                generator=torch_generator, device=device,
            )
            with torch.no_grad():
                reference_target = sampled_max_target(
                    target, reference, adapter.catalog_features, reference_candidates,
                    float(config["gamma"]),
                )
                weights = fixed_activity_weights(
                    reference["user_id"], fixed_activity_counts, activity_boundaries,
                    int(config["groups"]),
                )
            record.update(apply_gem_critic_step(
                critic, optimizer, loss, reference, reference_target, weights,
                float(config["min_group_ess"]),
            ))
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        profile_loss_value = update_profile(profile, profile_optimizer, batch, projection, config)
        soft_update(profile, teacher, float(config["teacher_tau"]))
        soft_update(critic, target, float(config["target_tau"]))
        if error_model is not None and error_target is not None:
            soft_update(error_model, error_target, float(config["target_tau"]))
        record["profile_loss"] = profile_loss_value
        records.append(record)
        if progress is not None and (
            update == 0 or (update + 1) % int(config["progress_interval"]) == 0
            or update + 1 == updates
        ):
            progress({
                "method": method, "update": update + 1, "updates": updates,
                "loss": record["loss"], "replay": len(replay),
                "reference_replay": len(reference_replay),
            })
    controlled = [entry for entry in records if entry["controlled"]]
    tail = records[-min(100, len(records)):]
    return critic.eval(), {
        "method": method,
        "updates": updates,
        "environment_interactions": updates * adapter.batch_size,
        "final_loss_mean": float(np.mean([entry["loss"] for entry in tail])),
        "final_profile_loss_mean": float(np.mean([entry["profile_loss"] for entry in tail])),
        "controlled_steps": len(controlled),
        "mean_norm_ratio": float(np.mean([entry["norm_ratio"] for entry in controlled])) if controlled else 1.0,
        "median_norm_ratio": float(np.median([entry["norm_ratio"] for entry in controlled])) if controlled else 1.0,
        "mean_cosine": float(np.mean([entry["cosine"] for entry in controlled])) if controlled else 1.0,
        "zero_update_rate": float(np.mean([entry["zero_update"] for entry in controlled])) if controlled else 0.0,
        "mean_active_groups": float(np.mean([entry.get("active_groups", 0) for entry in controlled])) if controlled else 0.0,
        "mean_protected_fraction": float(np.mean([entry.get("protected_fraction", 0) for entry in controlled])) if controlled else 0.0,
        "mean_discor_weight_ess": float(np.mean([
            entry["weight_ess"] for entry in records if "weight_ess" in entry
        ])) if method == "discor_adapted" else None,
        "records": records,
    }


def full_catalog_action_function(critic: QCritic, adapter: Any, chunk_size: int):
    def choose(state: torch.Tensor, observation: dict) -> torch.Tensor:
        return greedy_full_catalog(
            critic, state, adapter.catalog_features, int(chunk_size),
            forbidden=adapter.forbidden_actions(observation),
        )
    return choose


def equal_user_mean(values: list[float], users: list[int]) -> float:
    if len(values) != len(users) or not values:
        raise ValueError("values and users must be nonempty and aligned")
    per_user: dict[int, list[float]] = defaultdict(list)
    for user, value in zip(users, values):
        per_user[int(user)].append(float(value))
    return float(np.mean([np.mean(user_values) for user_values in per_user.values()]))


def evaluate_equal_user_complete_sessions(
    critic: QCritic,
    adapter: Any,
    encoded_users: np.ndarray,
    contexts_per_user: int,
    seed: int,
    horizon: int,
    discount: float,
    full_catalog_chunk: int,
    split: str = "test",
) -> dict[str, Any]:
    adapter.use_split(split)
    rows = adapter.uniform_user_context_rows(encoded_users, contexts_per_user, seed)
    action_fn = full_catalog_action_function(critic, adapter, full_catalog_chunk)
    totals: list[float] = []
    discounted: list[float] = []
    lengths: list[int] = []
    completed: list[bool] = []
    users: list[int] = []
    for start in range(0, len(rows), adapter.batch_size):
        actual_rows = rows[start:start + adapter.batch_size]
        actual_count = len(actual_rows)
        if actual_count < adapter.batch_size:
            padding = rows[:adapter.batch_size - actual_count]
            actual_rows = np.concatenate([actual_rows, padding])
        observation = adapter.observation_from_rows(actual_rows)
        torch.manual_seed(seed + 100_000 + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100_000 + start)
        result = adapter.evaluate_complete_context_batch(
            observation, action_fn, int(horizon), float(discount)
        )
        totals.extend(result["total_reward"][:actual_count].cpu().tolist())
        discounted.extend(result["discounted_return"][:actual_count].cpu().tolist())
        lengths.extend(result["episode_length"][:actual_count].cpu().tolist())
        completed.extend(result["completed"][:actual_count].cpu().tolist())
        users.extend(observation["user_profile"]["user_id"][:actual_count].cpu().tolist())
    completed_array = np.asarray(completed, dtype=bool)
    total_array = np.asarray(totals, dtype=np.float64)
    if not completed_array.all():
        raise RuntimeError(
            f"{int((~completed_array).sum())} evaluation sessions did not terminate by horizon {horizon}"
        )
    unique_users = len(set(map(int, users)))
    return {
        "avg_total_reward": equal_user_mean(totals, users),
        "episode_weighted_avg_total_reward": float(total_array.mean()),
        "avg_discounted_return": float(np.mean(discounted)),
        "avg_episode_length": float(np.mean(lengths)),
        "episodes": int(len(totals)),
        "users": int(unique_users),
        "contexts_per_user": int(contexts_per_user),
        "completion_rate": float(completed_array.mean()),
        "metric_definition": "equal-user mean of complete-session undiscounted cumulative reward",
        "episode_total_rewards": totals,
        "episode_lengths": lengths,
        "episode_user_ids": users,
    }
