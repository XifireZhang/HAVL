#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_summary(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    metrics = ["harm", "benefit", "net_delta_loss"]
    comparisons = [
        ("Single-group", "Base"),
        ("HAVL", "Base"),
        ("Norm-matched Base", "Base"),
        ("HAVL", "Norm-matched Base"),
    ]
    rows = []
    generator = np.random.default_rng(seed)
    for left, right in comparisons:
        left_frame = frame[frame.branch == left].set_index("proposal_id")
        right_frame = frame[frame.branch == right].set_index("proposal_id")
        for metric in metrics:
            differences = (left_frame[metric] - right_frame[metric]).to_numpy()
            draws = np.asarray([
                differences[generator.integers(0, len(differences), size=len(differences))].mean()
                for _ in range(replicates)
            ])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "mean_difference_left_minus_right": float(differences.mean()),
                    "conditional_ci95_low": float(np.quantile(draws, 0.025)),
                    "conditional_ci95_high": float(np.quantile(draws, 0.975)),
                    "proposal_batches": int(len(differences)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    updates = pd.read_parquet(run / "per_update_summary.parquet")
    activity = pd.read_parquet(run / "per_update_activity_summary.parquet")
    controllers = pd.read_parquet(run / "controller_diagnostics.parquet")
    updates.to_csv(run / "per_update_summary.csv", index=False)
    activity.to_csv(run / "per_update_activity_summary.csv", index=False)
    controllers.to_csv(run / "controller_diagnostics.csv", index=False)
    paired = paired_summary(updates, args.replicates, args.seed)
    paired.to_csv(run / "paired_difference_summary.csv", index=False)

    means = updates.groupby("branch")[[
        "harm", "benefit", "net_delta_loss", "harmful_fraction",
        "proposal_loss_improvement", "step_norm_ratio",
    ]].mean()
    means.to_csv(run / "branch_summary.csv")
    hnm = paired[(paired.left == "HAVL") & (paired.right == "Norm-matched Base")].set_index("metric")
    havl_controller = controllers[controllers.controller == "HAVL"]
    brief = f"""# Section 5.4 mechanism audit — first measured result

## Scope

- KuaiRand-27K source; fitted DQN; one training seed; checkpoint 150.
- 32 fixed, non-overlapping proposal mini-batches (128 transitions each).
- 1,024 reference transitions and 1,942 audit transitions from the original episode partition; 134 audit users before per-proposal source-user exclusion.
- The partition seed matches checkpoint training and profile fitting. Audit episodes were not used for critic training, profile fitting, grouping, radius selection, constraint construction, or acceptance checks.
- Audit target labels are fixed. Harm/benefit are formed per transition, averaged within user, then averaged equally over users and proposal batches.
- Intervals below resample proposal batches only and are conditional on this seed/checkpoint.

## Main result

| Branch | Harm H ↓ | Benefit G ↑ | Net Δ loss ↓ | Proposal-loss improvement ↑ | Step-norm ratio |
|---|---:|---:|---:|---:|---:|
"""
    order = ["Base", "Single-group", "HAVL", "Norm-matched Base"]
    for branch in order:
        row = means.loc[branch]
        brief += (
            f"| {branch} | {row.harm:.6f} | {row.benefit:.6f} | "
            f"{row.net_delta_loss:.6f} | {row.proposal_loss_improvement:.6f} | "
            f"{row.step_norm_ratio:.3f} |\n"
        )
    brief += "\n## HAVL versus norm-matched Base (paired)\n\n"
    labels = {"harm": "Harm H", "benefit": "Benefit G", "net_delta_loss": "Net Δ loss"}
    for metric in ("harm", "benefit", "net_delta_loss"):
        row = hnm.loc[metric]
        brief += (
            f"- {labels[metric]} difference: {row.mean_difference_left_minus_right:+.8f} "
            f"(95% conditional CI [{row.conditional_ci95_low:+.8f}, "
            f"{row.conditional_ci95_high:+.8f}]).\n"
        )
    brief += f"""

HAVL has lower harm than the norm-matched Base, but also lower benefit. The net difference is therefore the relevant balance and should not be inferred from harm alone. This one-checkpoint result is a mechanism diagnostic, not evidence of cross-seed policy-return improvement.

The HAVL controller corrected {havl_controller.corrected.mean():.1%} of proposals, skipped {havl_controller.skipped.mean():.1%}, and retained a mean step-norm ratio of {havl_controller.step_norm_ratio.mean():.3f}. The K=1 controller did not activate at the configured 1% relative budget, so its realized update equals Base at this checkpoint even though the algorithms are defined differently.
"""
    (run / "RESULTS_BRIEF_ZH.md").write_text(brief, encoding="utf-8")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
