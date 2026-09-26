#!/usr/bin/env python3
from __future__ import annotations

import argparse
from argparse import Namespace
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from bellman_sharing.data import stable_user_split
from bellman_sharing.kuaisim_adapter import NativeKuaiSimAdapter
from bellman_sharing.main_table import (
    evaluate_equal_user_complete_sessions,
    train_online_dqn,
)
from bellman_sharing.models import QCritic
from bellman_sharing.profile import orthogonal_projection
from bellman_sharing.profile import profile_targets
from bellman_sharing.training import evaluate_profile, train_profile_model


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def progress(value: dict) -> None:
    print(json.dumps({"time": now(), **value}), flush=True)


def source_manifest(config: dict, adapter: NativeKuaiSimAdapter) -> dict:
    kuaisim_root = Path(config["kuaisim"]["root"])
    model_log = kuaisim_root / config["kuaisim"]["model_log"]
    model_args = eval(model_log.read_text(encoding="utf-8").splitlines()[1])
    checkpoint = Path(model_args.model_path + ".checkpoint")
    if not checkpoint.is_absolute():
        checkpoint = kuaisim_root / "code" / checkpoint
    return {
        "created_at_utc": now(),
        "project_commit": git_commit(ROOT),
        "project_dirty": bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short", "--untracked-files=no"], text=True
        ).strip()),
        "kuaisim_commit": git_commit(kuaisim_root),
        "model_log": {"path": str(model_log), "sha256": sha256(model_log)},
        "model_checkpoint": {
            "path": str(checkpoint), "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "environment": adapter.metadata(),
        "scope_warning": (
            "KuaiRand-27K uses the existing 27K-source supported subset when the "
            "reported supported-user count is below the nominal 27K."
        ),
    }


def write_seed_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    key = (str(row["protocol"]), str(row["dataset"]), str(row["backbone"]),
           str(row["method"]), str(row["seed"]))
    rows = [existing for existing in rows if (
        existing["protocol"], existing["dataset"], existing["backbone"],
        existing["method"], existing["seed"]
    ) != key]
    rows.append({name: str(value) for name, value in row.items()})
    fields = list(row)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def constant_profile_metrics(replay, projection: torch.Tensor, gamma: float) -> dict[str, float]:
    reward = torch.as_tensor(replay.reward, dtype=torch.float32)
    continued = torch.as_tensor(replay.continued, dtype=torch.float32)
    next_state = torch.as_tensor(replay.next_state, dtype=torch.float32)
    target = profile_targets(reward, continued, next_state, projection.cpu(), gamma)
    return {
        "reward_mse": float((target["reward"] - target["reward"].mean()).square().mean()),
        "continuation_brier": float(
            (target["continued"] - target["continued"].mean()).square().mean()
        ),
        "successor_mse": float(
            (target["successor"] - target["successor"].mean(dim=0)).square().mean()
        ),
    }


def run(config_path: Path) -> None:
    config = load_config(config_path)
    run_id = config["run"]["run_id"]
    run_dir = ROOT / "runs/main_table_v1" / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True)
    shutil.copy2(config_path, run_dir / "config.toml")
    started = time.perf_counter()
    seed = int(config["run"]["seed"])
    run_kind = str(config["run"].get("kind", "development"))
    device = torch.device(config["run"]["device"])
    kuaisim_config = dict(config["kuaisim"])
    kuaisim_config["device"] = str(device)
    progress({"node": "load_environment", "run_id": run_id})
    adapter = NativeKuaiSimAdapter(kuaisim_config, seed)
    supported_users = np.arange(1, adapter.metadata()["supported_users"] + 1)
    splits = stable_user_split(
        supported_users,
        int(config["data"]["user_split_seed"]),
        float(config["data"]["train_user_fraction"]),
        float(config["data"]["validation_user_fraction"]),
    )
    adapter.configure_user_splits(splits)
    manifest = source_manifest(config, adapter)
    manifest["user_splits"] = {name: int(len(values)) for name, values in splits.items()}
    atomic_json(run_dir / "source_manifest.json", manifest)

    progress({"node": "collect_warmup"})
    warmup = adapter.collect(
        "train", int(config["data"]["warmup_stream_steps"]), seed + 100, 0
    )
    validation = adapter.collect(
        "validation", int(config["data"]["profile_validation_stream_steps"]),
        seed + 200, 100_000_000,
    )
    reference_mask = warmup.transition_id % int(config["dqn"]["reference_stride"]) == 0
    warmup_reference = warmup.take(np.flatnonzero(reference_mask))
    warmup_main = warmup.take(np.flatnonzero(~reference_mask))
    profile_config = dict(config["profile"])
    profile_config["gamma"] = float(config["kuaisim"]["gamma"])
    projection = orthogonal_projection(
        adapter.state_dim, int(profile_config["state_feature_dim"]), seed + 300
    ).to(device)
    progress({"node": "pretrain_profile", "records": len(warmup_main)})
    profile, profile_fit = train_profile_model(
        warmup_main, validation, projection, profile_config, device, seed + 400
    )
    profile_validation = evaluate_profile(profile, validation, projection, profile_config, device)
    profile_constant = constant_profile_metrics(
        validation, projection, float(config["kuaisim"]["gamma"])
    )
    profile_state = {name: value.detach().clone() for name, value in profile.state_dict().items()}
    torch.manual_seed(seed + 500)
    initial = QCritic(
        adapter.state_dim, adapter.action_feature_dim, config["dqn"]["hidden_dims"]
    ).to(device)
    initial_critic_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }
    del initial
    dqn_config = dict(config["dqn"])
    dqn_config.update({
        "gamma": float(config["kuaisim"]["gamma"]),
        "profile_hidden_dim": int(profile_config["hidden_dim"]),
        "profile_learning_rate": float(profile_config["learning_rate"]),
        "weight_reward": float(profile_config["weight_reward"]),
        "weight_continue": float(profile_config["weight_continue"]),
        "weight_successor": float(profile_config["weight_successor"]),
    })
    methods = list(config["run"]["methods"])
    freeze = {
        "protocol": "MAIN_TABLE_PROTOCOL_V1",
        "status": "frozen_before_method_training",
        "created_at_utc": now(),
        "run_id": run_id,
        "dataset": config["run"]["dataset"],
        "backbone": "dqn",
        "seed": seed,
        "run_kind": run_kind,
        "methods": methods,
        "config_sha256": sha256(config_path),
        "project_commit": git_commit(ROOT),
        "warmup": {"main": len(warmup_main), "reference": len(warmup_reference)},
        "profile_validation": profile_validation,
        "profile_constant_baseline": profile_constant,
        "candidate_support": {
            "behavior": "uniform full-catalog candidates, shared count across methods",
            "td_target": "uniform full-catalog candidates, shared count across methods",
            "evaluation": "greedy full-catalog scan with seen-item masking",
        },
        "primary_metric": "equal-user mean complete-session undiscounted cumulative reward",
    }
    atomic_json(run_dir / "freeze.json", freeze)
    results: dict[str, object] = {}
    for method_index, method in enumerate(methods):
        method_started = time.perf_counter()
        progress({"node": "train_method", "method": method})
        queue_record = {
            "protocol": "MAIN_TABLE_PROTOCOL_V1", "run_id": run_id,
            "dataset": config["run"]["dataset"], "backbone": "dqn",
            "method": method, "seed": seed, "status": "running",
            "run_kind": run_kind,
            "started_at_utc": now(), "config_sha256": sha256(config_path),
            "project_commit": git_commit(ROOT),
        }
        append_jsonl(ROOT / "reports/main_table_v1/run_manifest.jsonl", queue_record)
        try:
            critic, train_metrics = train_online_dqn(
                method, adapter, warmup_main, warmup_reference,
                initial_critic_state, profile_state, projection, dqn_config,
                seed + 1000, progress,
            )
            detail_records = train_metrics.pop("records")
            detail_path = run_dir / f"diagnostics_{method}.jsonl"
            with detail_path.open("w", encoding="utf-8") as handle:
                for record in detail_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            checkpoint_path = run_dir / "checkpoints" / f"dqn_{method}.pt"
            torch.save(critic.state_dict(), checkpoint_path)
            progress({"node": "evaluate_method", "method": method})
            evaluation_users = splits["test"]
            maximum_users = int(config["evaluation"].get("max_users", 0))
            if maximum_users > 0 and len(evaluation_users) > maximum_users:
                evaluation_users = np.random.default_rng(seed + 1999).choice(
                    evaluation_users, size=maximum_users, replace=False
                )
            evaluation = evaluate_equal_user_complete_sessions(
                critic, adapter, evaluation_users,
                int(config["evaluation"]["contexts_per_user"]), seed + 2000,
                int(config["evaluation"]["horizon"]),
                float(config["evaluation"]["discount"]),
                int(config["evaluation"]["full_catalog_chunk"]),
                split="test",
            )
            result = {
                "status": "complete", "method": method,
                "train": train_metrics, "evaluation": evaluation,
                "wall_seconds": time.perf_counter() - method_started,
                "checkpoint": {
                    "path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
                },
            }
            results[method] = result
            atomic_json(run_dir / "metrics.json", {
                "run_id": run_id, "profile_fit": profile_fit, "methods": results,
            })
            result_file = "seed_results.csv" if run_kind == "formal" else f"{run_kind}_results.csv"
            write_seed_result(ROOT / "reports/main_table_v1" / result_file, {
                "protocol": "MAIN_TABLE_PROTOCOL_V1",
                "dataset": config["run"]["dataset"], "backbone": "dqn",
                "method": method, "seed": seed,
                "avg_total_reward": evaluation["avg_total_reward"],
                "avg_episode_length": evaluation["avg_episode_length"],
                "episodes": evaluation["episodes"], "users": evaluation["users"],
                "completion_rate": evaluation["completion_rate"],
                "run_id": run_id, "status": "complete",
            })
            queue_record.update({
                "status": "complete", "ended_at_utc": now(),
                "result_path": str(run_dir / "metrics.json"),
                "avg_total_reward": evaluation["avg_total_reward"],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
            })
            progress({
                "node": "method_complete", "method": method,
                "avg_total_reward": evaluation["avg_total_reward"],
                "wall_seconds": result["wall_seconds"],
            })
        except Exception as error:
            queue_record.update({
                "status": "failed", "ended_at_utc": now(),
                "error_type": type(error).__name__, "error": str(error),
            })
            results[method] = dict(queue_record)
            atomic_json(run_dir / "metrics.json", {
                "run_id": run_id, "profile_fit": profile_fit, "methods": results,
            })
            append_jsonl(ROOT / "reports/main_table_v1/failures.jsonl", queue_record)
            progress({"node": "method_failed", "method": method, "error": repr(error)})
        append_jsonl(ROOT / "reports/main_table_v1/run_manifest.jsonl", queue_record)
        atomic_json(ROOT / "reports/main_table_v1/queue_status.json", {
            "updated_at_utc": now(), "active_run": run_id,
            "completed_methods": [name for name, value in results.items() if value["status"] == "complete"],
            "failed_methods": [name for name, value in results.items() if value["status"] == "failed"],
            "pending_methods": methods[method_index + 1:],
        })
    resources = {
        "wall_seconds": time.perf_counter() - started,
        "conservative_gpu_hours": (time.perf_counter() - started) / 3600,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "disk_free_bytes": shutil.disk_usage(run_dir).free,
    }
    atomic_json(run_dir / "manifest.json", {
        **manifest, "run_id": run_id, "run_kind": run_kind, "status": "complete",
        "profile_fit": profile_fit, "resources": resources,
        "methods": {name: value["status"] for name, value in results.items()},
    })
    progress({"status": "complete", "run_id": run_id, "resources": resources})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
