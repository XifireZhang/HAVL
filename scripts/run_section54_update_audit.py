#!/usr/bin/env python3
"""Paired one-step audit for Section 5.4 (Understanding Cross-User Sharing).

This is a mechanism diagnostic, not a policy-return experiment.  Every proposal
starts from one frozen learner snapshot.  Proposal, reference, and audit
transitions come from episode-disjoint replay partitions; audit receivers that
occur in the proposal batch are excluded from that proposal's measurements.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bellman_sharing.controller import (
    actual_optimizer_candidate,
    apply_with_loss_budgets,
    assign_flat,
    flat_gradient,
    project_candidate,
    trainable_parameters,
)
from bellman_sharing.data import Replay, assert_no_identity_overlap, tensor_batch
from bellman_sharing.experiments.episode_split import partition_replay_by_episode
from bellman_sharing.experiments.snapshot import restore_snapshot
from bellman_sharing.kuaisim_adapter import NativeKuaiSimAdapter
from bellman_sharing.models import BellmanProfile, QCritic
from bellman_sharing.profile import profile_vector
from bellman_sharing.training import sampled_target, weighted_losses


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_id(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:16]


def activity_labels(adapter: NativeKuaiSimAdapter, user_ids: np.ndarray) -> tuple[np.ndarray, list[float], np.ndarray]:
    """Predefine activity from the original KuaiRand interaction log."""
    lookup = {
        int(encoded): int(len(adapter.env.reader.user_history[raw]))
        for encoded, raw in adapter.encoded_to_raw_user.items()
    }
    counts = np.asarray(list(lookup.values()), dtype=np.int64)
    thresholds = np.quantile(counts, [1 / 3, 2 / 3])
    observed = np.asarray([lookup.get(int(user), 0) for user in user_ids], dtype=np.int64)
    labels = np.digitize(observed, thresholds, right=True).astype(np.int64)
    labels[observed == 0] = -1
    return labels, [float(x) for x in thresholds], observed


def controller_step(
    *,
    critic: QCritic,
    original: torch.Tensor,
    natural: torch.Tensor,
    reference: dict[str, torch.Tensor],
    reference_target: torch.Tensor,
    weights: torch.Tensor,
    active: torch.Tensor,
    relative_budget: float,
    trust_ratio: float,
    numerical_tolerance: float,
    backtracking_steps: int,
) -> tuple[torch.Tensor, dict]:
    assign_flat(trainable_parameters(critic), original)
    losses = weighted_losses(
        critic,
        reference["state"],
        reference["action_feature"],
        reference_target,
        weights,
        active,
    )
    gradients = torch.stack(
        [
            flat_gradient(loss, critic, retain_graph=slot + 1 < len(losses))
            for slot, loss in enumerate(losses)
        ]
    )
    before = losses.detach()
    allowed = relative_budget * before.clamp_min(0.0)
    natural_norm = float(torch.linalg.vector_norm(natural))
    projection = project_candidate(
        natural,
        gradients,
        allowed,
        trust_ratio * natural_norm,
    )

    def closure() -> torch.Tensor:
        return weighted_losses(
            critic,
            reference["state"],
            reference["action_feature"],
            reference_target,
            weights,
            active,
        )

    assign_flat(trainable_parameters(critic), original)
    accepted = apply_with_loss_budgets(
        critic,
        original,
        projection.displacement,
        closure,
        allowed,
        numerical_tolerance,
        backtracking_steps,
    )
    displacement = accepted.displacement.detach()
    accepted_norm = float(torch.linalg.vector_norm(displacement))
    correction_norm = float(torch.linalg.vector_norm(displacement - natural))
    return displacement, {
        "active_groups": int(len(active)),
        "natural_norm": natural_norm,
        "projected_norm": float(torch.linalg.vector_norm(projection.displacement)),
        "accepted_norm": accepted_norm,
        "step_norm_ratio": accepted_norm / max(natural_norm, 1e-24),
        "backtracking_scale": float(accepted.scale),
        "accepted": bool(accepted.accepted),
        "skipped": bool(accepted_norm <= 1e-24),
        "corrected": bool(correction_norm > 1e-8 * max(natural_norm, 1.0)),
        "projection_converged": bool(projection.converged),
        "maximum_linear_violation": float(projection.maximum_linear_violation),
        "reference_loss_before": before.detach().cpu().tolist(),
        "reference_loss_after": accepted.losses_after.detach().cpu().tolist(),
        "allowed_increase": allowed.detach().cpu().tolist(),
    }


def bootstrap_ci(update_frame: pd.DataFrame, value: str, replicates: int, seed: int) -> tuple[float, float]:
    pivot = update_frame.pivot(index="proposal_id", columns="branch", values=value)
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        draw = generator.integers(0, len(pivot), size=len(pivot))
        samples.append(pivot.iloc[draw].mean(axis=0).to_numpy())
    draws = np.asarray(samples)
    return draws


def plot_results(
    update_summary: pd.DataFrame,
    activity_summary: pd.DataFrame,
    output: Path,
    replicates: int,
    seed: int,
    audit_samples: int,
    audit_users: int,
) -> None:
    order = ["Base", "Single-group", "HAVL", "Norm-matched Base"]
    colors = ["#808080", "#D95F02", "#1B9E77", "#7570B3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ["harm", "benefit"],
        ["Cross-user harm (lower is better)", "Cross-user benefit (higher is better)"],
    ):
        pivot = update_summary.pivot(index="proposal_id", columns="branch", values=metric)[order]
        generator = np.random.default_rng(seed + (0 if metric == "harm" else 1))
        draws = np.asarray([
            pivot.iloc[generator.integers(0, len(pivot), size=len(pivot))].mean(axis=0).to_numpy()
            for _ in range(replicates)
        ])
        mean = pivot.mean(axis=0).to_numpy()
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        positions = np.arange(len(order))
        axis.bar(positions, mean, color=colors, width=0.72)
        axis.errorbar(positions, mean, yerr=[mean - low, high - mean], fmt="none", color="black", capsize=3, lw=1)
        axis.set_xticks(positions, ["Base", "K=1", "HAVL", "Norm-match"], rotation=15)
        axis.set_ylabel(
            "Mean positive TD-loss change"
            if metric == "harm"
            else "Mean negative TD-loss change magnitude"
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"KuaiRand-27K source · DQN · 1 checkpoint · 32 paired proposals\n"
        f"Initial audit pool: {audit_users:,} users / {audit_samples:,} contexts · "
        "95% conditional CI over proposals",
        fontsize=10,
    )
    fig.savefig(output / "paired_harm_benefit.png", dpi=220)
    fig.savefig(output / "paired_harm_benefit.pdf")
    plt.close(fig)

    labels = {0: "Low", 1: "Medium", 2: "High", -1: "Unseen"}
    kept = activity_summary[activity_summary["activity_stratum"] >= 0].copy()
    aggregate = kept.groupby(["branch", "activity_stratum"], as_index=False)["harm"].mean()
    fig, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    width = 0.19
    x = np.arange(3)
    for slot, (branch, color) in enumerate(zip(order, colors)):
        frame = aggregate[aggregate["branch"] == branch].set_index("activity_stratum")
        values = [float(frame.loc[level, "harm"]) if level in frame.index else np.nan for level in range(3)]
        axis.bar(x + (slot - 1.5) * width, values, width=width, color=color, label=branch)
    axis.set_xticks(x, [labels[level] for level in range(3)])
    axis.set_xlabel("User interaction-count stratum (predefined from raw KuaiRand log)")
    axis.set_ylabel("Mean positive TD-loss change")
    axis.set_title("Cross-user harm by user contribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.text(0.5, 0.01, "KuaiRand-27K source · DQN · 1 checkpoint", ha="center", fontsize=8)
    fig.savefig(output / "harm_by_activity.png", dpi=220)
    fig.savefig(output / "harm_by_activity.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_toml(config_path)
    source = (ROOT / config["run"]["source_run"]).resolve()
    profile_run = (ROOT / config["run"]["profile_run"]).resolve()
    snapshot_path = (ROOT / config["run"]["snapshot"]).resolve()
    output = (ROOT / config["run"]["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.toml")

    seed = int(config["run"]["seed"])
    settings = config["audit"]
    source_config = load_toml(source / "config.toml")
    profile_config = load_toml(profile_run / "config.toml")
    device = torch.device(source_config["run"]["device"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = Replay.load(source / "replay_train.npz")
    partitions, episode_roles = partition_replay_by_episode(
        train,
        int(config["data"]["episode_batch_size"]),
        int(config["data"]["partition_seed"]),
        float(config["data"]["training_fraction"]),
        float(config["data"]["reference_fraction"]),
    )
    assert_no_identity_overlap(partitions["training"], partitions["reference"], partitions["audit"])

    kuaisim_config = dict(source_config["kuaisim"])
    kuaisim_config["device"] = str(device)
    adapter = NativeKuaiSimAdapter(kuaisim_config, seed + 41)
    catalog = adapter.catalog_features
    dqn = source_config["dqn"]
    critic = QCritic(train.state.shape[1], train.action_feature.shape[1], dqn["hidden_dims"]).to(device)
    target = QCritic(train.state.shape[1], train.action_feature.shape[1], dqn["hidden_dims"]).to(device)
    profile = BellmanProfile(
        train.state.shape[1],
        train.action_feature.shape[1],
        int(profile_config["profile"]["state_feature_dim"]),
        int(profile_config["profile"]["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(dqn["learning_rate"]),
        weight_decay=float(dqn["weight_decay"]),
    )
    # Keep serialized CPU/CUDA RNG byte tensors on CPU.  Moving the complete
    # snapshot to CUDA makes torch.set_rng_state reject its CPU RNG state.
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    extras = restore_snapshot(
        snapshot,
        {"critic": critic, "target": target, "profile": profile},
        {"critic": optimizer},
    )
    grouping = deepcopy(extras["grouping"])
    grouping.prototypes = grouping.prototypes.to(device)
    target.eval()
    profile.eval()

    generator = np.random.default_rng(seed + 991)
    proposal_count = int(settings["proposal_count"])
    proposal_size = int(settings["proposal_batch_size"])
    proposal_order = generator.permutation(len(partitions["training"]))[: proposal_count * proposal_size]
    proposal_batches = proposal_order.reshape(proposal_count, proposal_size)
    reference_size = min(int(settings["reference_batch_size"]), len(partitions["reference"]))
    audit_size = min(int(settings["audit_batch_size"]), len(partitions["audit"]))
    reference_indices = generator.choice(
        len(partitions["reference"]), size=reference_size, replace=False
    )
    audit_indices = generator.choice(
        len(partitions["audit"]), size=audit_size, replace=False
    )
    candidates = generator.choice(
        len(catalog), size=int(settings["target_candidate_count"]), replace=False
    )
    reference = tensor_batch(partitions["reference"], reference_indices, device)
    audit = tensor_batch(partitions["audit"], audit_indices, device)
    gamma = float(source_config["kuaisim"]["gamma"])
    with torch.no_grad():
        reference_target = sampled_target(
            target, reference["next_state"], reference["reward"], reference["continued"], catalog, candidates, gamma
        ).detach()
        audit_target = sampled_target(
            target, audit["next_state"], audit["reward"], audit["continued"], catalog, candidates, gamma
        ).detach()
        audit_before = 0.5 * (
            critic(audit["state"], audit["action_feature"]) - audit_target
        ).square()
        reference_profile = profile_vector(profile, reference["state"], reference["action_feature"])
        havl_weights, reference_ungrouped = grouping.weights(reference_profile)
        havl_weights = havl_weights.detach()
        havl_mass = havl_weights.sum(dim=0)
        havl_active = torch.nonzero(
            havl_mass >= float(settings["min_group_weight"]), as_tuple=False
        ).flatten()
        single_weights = torch.ones((len(reference_indices), 1), device=device)
        single_active = torch.tensor([0], dtype=torch.long, device=device)

    audit_user_ids = audit["user_id"].detach().cpu().numpy()
    activity, activity_thresholds, interaction_counts = activity_labels(adapter, audit_user_ids)
    raw_records: list[dict] = []
    update_records: list[dict] = []
    activity_records: list[dict] = []
    controller_records: list[dict] = []

    for proposal_id, proposal_indices in enumerate(proposal_batches):
        restore_snapshot(
            snapshot,
            {"critic": critic, "target": target, "profile": profile},
            {"critic": optimizer},
        )
        proposal = tensor_batch(partitions["training"], proposal_indices, device)
        with torch.no_grad():
            proposal_target = sampled_target(
                target,
                proposal["next_state"],
                proposal["reward"],
                proposal["continued"],
                catalog,
                candidates,
                gamma,
            ).detach()
        proposal_loss = F.mse_loss(
            critic(proposal["state"], proposal["action_feature"]), proposal_target
        )
        original, natural = actual_optimizer_candidate(critic, optimizer, proposal_loss)
        assign_flat(trainable_parameters(critic), original)

        single_step, single_diag = controller_step(
            critic=critic,
            original=original,
            natural=natural,
            reference=reference,
            reference_target=reference_target,
            weights=single_weights,
            active=single_active,
            relative_budget=float(settings["relative_loss_budget"]),
            trust_ratio=float(settings["trust_radius_ratio"]),
            numerical_tolerance=float(settings["numerical_tolerance"]),
            backtracking_steps=int(settings["backtracking_steps"]),
        )
        havl_step, havl_diag = controller_step(
            critic=critic,
            original=original,
            natural=natural,
            reference=reference,
            reference_target=reference_target,
            weights=havl_weights,
            active=havl_active,
            relative_budget=float(settings["relative_loss_budget"]),
            trust_ratio=float(settings["trust_radius_ratio"]),
            numerical_tolerance=float(settings["numerical_tolerance"]),
            backtracking_steps=int(settings["backtracking_steps"]),
        )
        natural_norm = float(torch.linalg.vector_norm(natural))
        havl_norm = float(torch.linalg.vector_norm(havl_step))
        norm_matched = natural * (havl_norm / natural_norm) if natural_norm > 1e-24 else torch.zeros_like(natural)
        branches = {
            "Base": natural.detach(),
            "Single-group": single_step,
            "HAVL": havl_step,
            "Norm-matched Base": norm_matched.detach(),
        }
        proposal_users = np.unique(proposal["user_id"].detach().cpu().numpy())
        receiver_mask_np = ~np.isin(audit_user_ids, proposal_users)
        receiver_mask = torch.as_tensor(receiver_mask_np, device=device)
        receiver_count = int(receiver_mask_np.sum())
        if receiver_count == 0:
            raise RuntimeError("no user-disjoint audit receivers remain")

        for controller_name, diagnostic in (("Single-group", single_diag), ("HAVL", havl_diag)):
            controller_records.append({"proposal_id": proposal_id, "controller": controller_name, **diagnostic})

        for branch, displacement in branches.items():
            assign_flat(trainable_parameters(critic), original + displacement)
            with torch.no_grad():
                audit_after = 0.5 * (
                    critic(audit["state"], audit["action_feature"]) - audit_target
                ).square()
                change = (audit_after - audit_before).detach().cpu().numpy()
                train_after = F.mse_loss(
                    critic(proposal["state"], proposal["action_feature"]), proposal_target
                )
            selected = change[receiver_mask_np]
            selected_users = audit_user_ids[receiver_mask_np]
            receiver_table = pd.DataFrame(
                {
                    "user_id": selected_users,
                    "delta": selected,
                    "harm": np.maximum(selected, 0.0),
                    "benefit": np.maximum(-selected, 0.0),
                    "harmful": (selected > 0).astype(np.float64),
                }
            )
            # A user with many audit transitions must not dominate the metric.
            receiver_user_metrics = receiver_table.groupby("user_id", as_index=False).agg(
                harm=("harm", "mean"),
                benefit=("benefit", "mean"),
                net_delta_loss=("delta", "mean"),
                harmful_fraction=("harmful", "mean"),
            )
            branch_norm = float(torch.linalg.vector_norm(displacement))
            update_records.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_batch_id": array_id(proposal_indices),
                    "branch": branch,
                    "receiver_samples": receiver_count,
                    "receiver_users": int(len(receiver_user_metrics)),
                    "proposal_users": int(len(proposal_users)),
                    "harm": float(receiver_user_metrics["harm"].mean()),
                    "benefit": float(receiver_user_metrics["benefit"].mean()),
                    "net_delta_loss": float(receiver_user_metrics["net_delta_loss"].mean()),
                    "harmful_fraction": float(receiver_user_metrics["harmful_fraction"].mean()),
                    "proposal_loss_before": float(proposal_loss.detach()),
                    "proposal_loss_after": float(train_after.detach()),
                    "proposal_loss_improvement": float(proposal_loss.detach() - train_after.detach()),
                    "accepted_norm": branch_norm,
                    "step_norm_ratio": branch_norm / max(natural_norm, 1e-24),
                }
            )
            for stratum in (-1, 0, 1, 2):
                mask = receiver_mask_np & (activity == stratum)
                if not mask.any():
                    continue
                values = change[mask]
                stratum_users = audit_user_ids[mask]
                stratum_table = pd.DataFrame(
                    {
                        "user_id": stratum_users,
                        "delta": values,
                        "harm": np.maximum(values, 0.0),
                        "benefit": np.maximum(-values, 0.0),
                        "harmful": (values > 0).astype(np.float64),
                    }
                ).groupby("user_id", as_index=False).agg(
                    harm=("harm", "mean"),
                    benefit=("benefit", "mean"),
                    net_delta_loss=("delta", "mean"),
                    harmful_fraction=("harmful", "mean"),
                )
                activity_records.append(
                    {
                        "proposal_id": proposal_id,
                        "branch": branch,
                        "activity_stratum": stratum,
                        "samples": int(mask.sum()),
                        "users": int(len(stratum_table)),
                        "harm": float(stratum_table["harm"].mean()),
                        "benefit": float(stratum_table["benefit"].mean()),
                        "net_delta_loss": float(stratum_table["net_delta_loss"].mean()),
                        "harmful_fraction": float(stratum_table["harmful_fraction"].mean()),
                    }
                )
            selected_indices = np.flatnonzero(receiver_mask_np)
            for index in selected_indices:
                raw_records.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_batch_id": array_id(proposal_indices),
                        "branch": branch,
                        "audit_transition_id": int(audit["transition_id"][index].detach().cpu()),
                        "receiver_user_id": int(audit_user_ids[index]),
                        "activity_stratum": int(activity[index]),
                        "training_interaction_count": int(interaction_counts[index]),
                        "delta_td_loss": float(change[index]),
                        "harm": float(max(change[index], 0.0)),
                        "benefit": float(max(-change[index], 0.0)),
                    }
                )

    assign_flat(trainable_parameters(critic), original)
    raw = pd.DataFrame(raw_records)
    updates = pd.DataFrame(update_records)
    activity_frame = pd.DataFrame(activity_records)
    controllers = pd.DataFrame(controller_records)
    raw.to_parquet(output / "per_update_user_audit.parquet", index=False)
    updates.to_parquet(output / "per_update_summary.parquet", index=False)
    activity_frame.to_parquet(output / "per_update_activity_summary.parquet", index=False)
    controllers.to_parquet(output / "controller_diagnostics.parquet", index=False)

    order = ["Base", "Single-group", "HAVL", "Norm-matched Base"]
    summary: dict[str, dict] = {}
    for branch in order:
        frame = updates[updates["branch"] == branch]
        summary[branch] = {}
        for metric in ("harm", "benefit", "net_delta_loss", "harmful_fraction", "proposal_loss_improvement", "step_norm_ratio"):
            values = frame[metric].to_numpy()
            generator_ci = np.random.default_rng(seed + len(summary) * 37 + len(metric))
            draws = np.asarray([
                values[generator_ci.integers(0, len(values), size=len(values))].mean()
                for _ in range(int(settings["bootstrap_replicates"]))
            ])
            summary[branch][metric] = {
                "mean": float(values.mean()),
                "conditional_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            }
    base = updates[updates["branch"] == "Base"].set_index("proposal_id")
    paired = {}
    for branch in order[1:]:
        frame = updates[updates["branch"] == branch].set_index("proposal_id")
        paired[branch] = {
            "harm_minus_base": float((frame["harm"] - base["harm"]).mean()),
            "benefit_minus_base": float((frame["benefit"] - base["benefit"]).mean()),
            "net_delta_loss_minus_base": float((frame["net_delta_loss"] - base["net_delta_loss"]).mean()),
        }
    controller_summary = controllers.groupby("controller").agg(
        correction_rate=("corrected", "mean"),
        skip_rate=("skipped", "mean"),
        mean_step_norm_ratio=("step_norm_ratio", "mean"),
        mean_backtracking_scale=("backtracking_scale", "mean"),
    ).to_dict(orient="index")

    summary_payload = {
        "scope": "KuaiRand-27K source, fitted DQN, one training seed, one checkpoint",
        "checkpoint_step": int(extras.get("checkpoint_step", -1)),
        "proposal_batches": proposal_count,
        "proposal_batch_size": proposal_size,
        "reference_samples": int(len(reference_indices)),
        "audit_samples_before_user_exclusion": int(len(audit_indices)),
        "audit_users_before_user_exclusion": int(np.unique(audit_user_ids).size),
        "audit_identity": "episode-disjoint held-out transitions; proposal-user rows removed per update",
        "aggregation": "positive/negative parts are computed per transition, averaged within user, then averaged equally across users and proposal batches",
        "activity_source": "original KuaiRand log interaction count from the native reader user history",
        "activity_thresholds_interaction_count": activity_thresholds,
        "havl_active_groups": int(len(havl_active)),
        "havl_reference_group_mass": havl_mass.detach().cpu().tolist(),
        "havl_reference_ungrouped_fraction": float(reference_ungrouped.float().mean()),
        "branch_metrics": summary,
        "paired_differences": paired,
        "controller_metrics": controller_summary,
        "uncertainty_boundary": "95% bootstrap intervals resample the 32 fixed proposal batches. They are conditional on one seed/checkpoint and are not training-seed uncertainty.",
    }
    (output / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    plot_results(
        updates,
        activity_frame,
        output,
        int(settings["bootstrap_replicates"]),
        seed,
        int(len(audit_indices)),
        int(np.unique(audit_user_ids).size),
    )

    manifest = {
        "experiment_id": config["run"]["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
        "config_sha256": file_sha256(config_path),
        "snapshot": str(snapshot_path),
        "snapshot_sha256": file_sha256(snapshot_path),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "profile_checkpoint_sha256": file_sha256(profile_run / "profile.pt"),
        "reference_transition_ids_sha256": array_id(partitions["reference"].transition_id[reference_indices]),
        "audit_transition_ids_sha256": array_id(partitions["audit"].transition_id[audit_indices]),
        "proposal_transition_ids_sha256": array_id(partitions["training"].transition_id[proposal_order]),
        "target_candidate_indices_sha256": array_id(candidates),
        "device": str(device),
        "data_roles": {
            name: {"transitions": len(replay), "episodes": int(len(episode_roles[name]))}
            for name, replay in partitions.items()
        },
        "fixed_target_labels": True,
        "audit_used_for_grouping_tuning_or_acceptance": False,
        "proposal_reference_audit_transition_disjoint": True,
        "receiver_user_disjoint_enforced_per_proposal": True,
        "partition_seed": int(config["data"]["partition_seed"]),
        "partition_matches_checkpoint_training": True,
        "audit_used_for_checkpoint_training": False,
        "audit_used_for_profile_fitting": False,
        "audit_used_for_grouping_or_radius_selection": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
