from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .controller import (
    actual_optimizer_candidate,
    apply_with_backtracking,
    flat_gradient,
    project_candidate,
    trainable_parameters,
)
from .data import Replay, tensor_batch
from .models import BellmanProfile, QCritic, greedy_full_catalog, max_sampled_q
from .profile import (
    SoftGrouping,
    fit_grouping,
    profile_loss,
    profile_targets,
    profile_vector,
)


def seeded_indices(length: int, count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, length, size=count, dtype=np.int64)


def train_profile_model(
    train: Replay,
    validation: Replay,
    projection: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[BellmanProfile, dict[str, Any]]:
    torch.manual_seed(seed)
    model = BellmanProfile(
        train.state.shape[1], train.action_feature.shape[1], projection.shape[1],
        int(config["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    weights = (
        float(config["weight_reward"]), float(config["weight_continue"]),
        float(config["weight_successor"]),
    )
    generator = np.random.default_rng(seed + 1)
    steps_per_epoch = math.ceil(len(train) / int(config["batch_size"]))
    losses = []
    for epoch in range(int(config["epochs"])):
        order = generator.permutation(len(train))
        for slot in range(steps_per_epoch):
            indices = order[slot * int(config["batch_size"]):(slot + 1) * int(config["batch_size"])]
            batch = tensor_batch(train, indices, device)
            target = profile_targets(
                batch["reward"], batch["continued"], batch["next_state"], projection,
                float(config["gamma"]),
            )
            prediction = model(batch["state"], batch["action_feature"])
            loss, components = profile_loss(prediction, target, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    validation_metrics = evaluate_profile(model, validation, projection, config, device)
    return model.eval(), {
        "train_final_loss_mean": float(np.mean(losses[-steps_per_epoch:])),
        "epochs": int(config["epochs"]),
        "validation": validation_metrics,
    }


@torch.no_grad()
def evaluate_profile(
    model: BellmanProfile,
    replay: Replay,
    projection: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    predictions, targets = [], []
    for start in range(0, len(replay), batch_size):
        batch = tensor_batch(replay, np.arange(start, min(start + batch_size, len(replay))), device)
        output = model(batch["state"], batch["action_feature"])
        target = profile_targets(
            batch["reward"], batch["continued"], batch["next_state"], projection,
            float(config["gamma"]),
        )
        predictions.append({name: value.detach().cpu() for name, value in output.items()})
        targets.append({name: value.detach().cpu() for name, value in target.items()})
    prediction = {name: torch.cat([block[name] for block in predictions]) for name in predictions[0]}
    target = {name: torch.cat([block[name] for block in targets]) for name in targets[0]}
    probability = torch.sigmoid(prediction["continue_logit"])
    return {
        "records": len(replay),
        "users": int(np.unique(replay.user_id).size),
        "reward_mse": float(F.mse_loss(prediction["reward"], target["reward"])),
        "continuation_brier": float(F.mse_loss(probability, target["continued"])),
        "continuation_bce": float(F.binary_cross_entropy(probability, target["continued"])),
        "successor_mse": float(F.mse_loss(prediction["successor"], target["successor"])),
        "continuation_rate": float(target["continued"].mean()),
    }


@torch.no_grad()
def all_profile_vectors(
    model: BellmanProfile, replay: Replay, device: torch.device, batch_size: int = 1024
) -> torch.Tensor:
    output = []
    for start in range(0, len(replay), batch_size):
        batch = tensor_batch(replay, np.arange(start, min(start + batch_size, len(replay))), device)
        output.append(profile_vector(model, batch["state"], batch["action_feature"]))
    return torch.cat(output)


def build_profile_grouping(
    model: BellmanProfile,
    reference: Replay,
    validation: Replay,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[SoftGrouping, torch.Tensor, dict[str, Any]]:
    reference_vectors = all_profile_vectors(model, reference, device)
    validation_vectors = all_profile_vectors(model, validation, device)
    grouping = fit_grouping(
        reference_vectors, validation_vectors,
        int(config["prototype_count"]), float(config["radius_quantile"]),
        float(config["soft_temperature"]), float(config["gamma"]),
        float(config["prototype_ema"]),
    )
    weights, ungrouped = grouping.weights(reference_vectors)
    return grouping, reference_vectors.detach(), {
        "radius": grouping.radius,
        "scales": list(grouping.scales),
        "reference_ungrouped_fraction": float(ungrouped.float().mean()),
        "reference_group_mass": weights.sum(dim=0).detach().cpu().tolist(),
    }


def sampled_target(
    critic_target: QCritic,
    next_state: torch.Tensor,
    reward: torch.Tensor,
    continued: torch.Tensor,
    catalog_features: torch.Tensor,
    candidate_indices: np.ndarray,
    gamma: float,
) -> torch.Tensor:
    indices = torch.as_tensor(candidate_indices, dtype=torch.long, device=next_state.device)
    candidate = catalog_features[indices][None, :, :].expand(len(next_state), -1, -1)
    return reward + gamma * continued * max_sampled_q(critic_target, next_state, candidate)


def group_weights(
    method: str,
    batch: dict[str, torch.Tensor],
    profile_vectors: torch.Tensor,
    grouping: SoftGrouping,
    target_values: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "profile":
        return grouping.weights(profile_vectors)
    if method == "value":
        boundaries = torch.quantile(
            target_values.detach(), torch.linspace(0, 1, groups + 1, device=target_values.device)
        )[1:-1]
        assignment = torch.bucketize(target_values.detach(), boundaries)
    elif method == "random":
        assignment = torch.remainder(batch["transition_id"], groups)
    else:
        raise ValueError(method)
    weights = F.one_hot(assignment, num_classes=groups).float()
    return weights.detach(), torch.zeros(len(weights), dtype=torch.bool, device=weights.device)


def weighted_losses(
    critic: QCritic,
    state: torch.Tensor,
    action_feature: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    active_groups: torch.Tensor,
) -> torch.Tensor:
    per_sample = 0.5 * (critic(state, action_feature) - target) ** 2
    output = []
    for group in active_groups:
        weight = weights[:, group]
        output.append((weight * per_sample).sum() / weight.sum().clamp_min(1e-12))
    return torch.stack(output) if output else torch.empty(0, device=state.device)


@dataclass
class MethodRun:
    critic: QCritic
    target: QCritic
    metrics: dict[str, Any]
    norm_schedule: list[float]


def train_dqn_method(
    method: str,
    initial_state: dict[str, torch.Tensor],
    base: Replay,
    reference: Replay,
    audit: Replay,
    catalog_features: torch.Tensor,
    profile_model: BellmanProfile,
    grouping_template: SoftGrouping,
    reference_profile_vectors: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    matched_norm_schedule: list[float] | None = None,
) -> MethodRun:
    torch.manual_seed(seed)
    critic = QCritic(base.state.shape[1], base.action_feature.shape[1], config["hidden_dims"]).to(device)
    critic.load_state_dict(initial_state)
    target = deepcopy(critic).eval()
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    grouping = deepcopy(grouping_template)
    generator = np.random.default_rng(seed + 13)
    update_batches = generator.integers(0, len(base), size=(int(config["updates"]), int(config["batch_size"])))
    reference_batches = generator.integers(
        0, len(reference), size=(int(config["updates"]), int(config["reference_batch_size"]))
    )
    target_candidates = generator.integers(
        0, len(catalog_features), size=(int(config["updates"]), int(config["target_candidate_count"])),
    )
    reference_candidates = generator.integers(
        0, len(catalog_features), size=(int(config["updates"]), int(config["target_candidate_count"])),
    )
    records: list[dict[str, Any]] = []
    norm_schedule: list[float] = []
    for update in range(int(config["updates"])):
        batch = tensor_batch(base, update_batches[update], device)
        with torch.no_grad():
            y = sampled_target(
                target, batch["next_state"], batch["reward"], batch["continued"],
                catalog_features, target_candidates[update], float(config["gamma"]),
            )
        loss = F.mse_loss(critic(batch["state"], batch["action_feature"]), y)
        original, natural = actual_optimizer_candidate(critic, optimizer, loss)
        natural_norm = float(torch.linalg.vector_norm(natural))
        applied = natural
        entry: dict[str, Any] = {
            "update": update, "base_loss": float(loss.detach()), "controlled": False,
            "natural_norm": natural_norm, "norm_ratio": 1.0, "cosine": 1.0,
            "backtracking_scale": 1.0, "zero_update": False,
        }
        controlled = update % int(config["control_interval"]) == 0
        if method in {"profile", "value", "random"} and controlled:
            ref_indices = reference_batches[update]
            ref = tensor_batch(reference, ref_indices, device)
            with torch.no_grad():
                ref_target = sampled_target(
                    target, ref["next_state"], ref["reward"], ref["continued"],
                    catalog_features, reference_candidates[update], float(config["gamma"]),
                )
                ref_profile = reference_profile_vectors[torch.as_tensor(ref_indices, device=device)]
                values = target(ref["state"], ref["action_feature"])
                weights, ungrouped = group_weights(
                    method, ref, ref_profile, grouping, values,
                    int(config["prototype_count"]),
                )
                mass = weights.sum(dim=0)
                active = torch.nonzero(
                    mass >= float(config["min_group_weight"]), as_tuple=False
                ).flatten()
            losses = weighted_losses(
                critic, ref["state"], ref["action_feature"], ref_target, weights, active
            )
            gradients = torch.stack([
                flat_gradient(value, critic, retain_graph=slot + 1 < len(losses))
                for slot, value in enumerate(losses)
            ]) if len(losses) else torch.empty((0, natural.numel()), device=device)
            radius = float(config["trust_radius_ratio"]) * natural_norm
            projection = project_candidate(
                natural, gradients,
                torch.full((len(gradients),), float(config["linear_budget"]), device=device),
                radius,
            )

            def closure() -> torch.Tensor:
                return weighted_losses(
                    critic, ref["state"], ref["action_feature"], ref_target, weights, active
                )

            backtracking = apply_with_backtracking(
                critic, original, projection.displacement, closure,
                float(config["actual_loss_tolerance"]), int(config["backtracking_steps"]),
            )
            applied = backtracking.displacement
            if method == "profile":
                grouping.slow_update(ref_profile, weights)
            applied_norm = float(torch.linalg.vector_norm(applied))
            cosine = float(torch.dot(applied, natural) / max(applied_norm * natural_norm, 1e-24))
            entry.update({
                "controlled": True, "active_groups": int(len(active)),
                "ungrouped_fraction": float(ungrouped.float().mean()),
                "norm_ratio": applied_norm / max(natural_norm, 1e-24), "cosine": cosine,
                "backtracking_scale": backtracking.scale,
                "zero_update": not backtracking.accepted,
                "maximum_linear_violation": projection.maximum_linear_violation,
                "projection_converged": projection.converged,
                "max_actual_loss_change": float((backtracking.losses_after - backtracking.losses_before).max()) if len(active) else 0.0,
            })
        elif method == "matched" and controlled:
            if matched_norm_schedule is None:
                raise ValueError("matched method requires a frozen norm schedule")
            ratio = matched_norm_schedule[len(norm_schedule)]
            applied = ratio * natural
            entry.update({"controlled": True, "norm_ratio": ratio, "cosine": 1.0})
        from .controller import assign_flat
        assign_flat(trainable_parameters(critic), original + applied)
        ratio = float(torch.linalg.vector_norm(applied)) / max(natural_norm, 1e-24)
        if controlled:
            norm_schedule.append(ratio)
        with torch.no_grad():
            tau = float(config["target_tau"])
            for parameter, target_parameter in zip(critic.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, tau)
        records.append(entry)
    audit_metrics = audit_td_metrics(
        critic, target, audit, catalog_features, config, device, seed + 99
    )
    controlled_records = [record for record in records if record["controlled"]]
    metrics = {
        "method": method,
        "updates": int(config["updates"]),
        "final_base_loss_mean": float(np.mean([r["base_loss"] for r in records[-20:]])),
        "controlled_steps": len(controlled_records),
        "correction_rate": float(np.mean([r["norm_ratio"] < .999999 for r in controlled_records])) if controlled_records else 0.0,
        "zero_update_rate": float(np.mean([r["zero_update"] for r in controlled_records])) if controlled_records else 0.0,
        "mean_norm_ratio": float(np.mean([r["norm_ratio"] for r in controlled_records])) if controlled_records else 1.0,
        "mean_cosine": float(np.mean([r["cosine"] for r in controlled_records])) if controlled_records else 1.0,
        "mean_backtracking_scale": float(np.mean([r["backtracking_scale"] for r in controlled_records])) if controlled_records else 1.0,
        "projection_failure_count": int(sum(not r.get("projection_converged", True) for r in controlled_records)),
        "audit": audit_metrics,
        "records": records,
    }
    return MethodRun(critic.eval(), target.eval(), metrics, norm_schedule)


@torch.no_grad()
def audit_td_metrics(
    critic: QCritic,
    target: QCritic,
    audit: Replay,
    catalog_features: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    losses, values = [], []
    for start in range(0, len(audit), int(config["batch_size"])):
        indices = np.arange(start, min(start + int(config["batch_size"]), len(audit)))
        batch = tensor_batch(audit, indices, device)
        candidates = generator.integers(0, len(catalog_features), size=int(config["target_candidate_count"]))
        y = sampled_target(
            target, batch["next_state"], batch["reward"], batch["continued"],
            catalog_features, candidates, float(config["gamma"]),
        )
        q = critic(batch["state"], batch["action_feature"])
        losses.extend(((q - y) ** 2).cpu().tolist())
        values.extend(q.cpu().tolist())
    return {
        "td_mse": float(np.mean(losses)),
        "q_mean": float(np.mean(values)),
        "q_std": float(np.std(values)),
    }


def policy_action_function(critic: QCritic, adapter: Any, chunk_size: int):
    def choose(state: torch.Tensor, observation: dict) -> torch.Tensor:
        return greedy_full_catalog(
            critic, state, adapter.catalog_features, chunk_size,
            forbidden=adapter.forbidden_actions(observation),
        )
    return choose
