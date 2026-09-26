#!/usr/bin/env python3
"""Plot the existing M2 cache as a distance-ranking diagnostic (not grouping)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.run_dir / "coverage_backup_curve.parquet")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "distance_ranking_coverage.csv", index=False)
    labels = {
        "current_value_distance": "Current value",
        "current_embedding_distance": "Critic embedding",
        "state_distance": "HAVL descriptor",
        "bellman_profile_distance": "Raw state",
    }
    colors = {
        "current_value_distance": "#D95F02",
        "current_embedding_distance": "#1B9E77",
        "state_distance": "#7570B3",
        "bellman_profile_distance": "#4C78A8",
    }
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.23, top=0.88)
    for name, group in frame.groupby("distance", sort=False):
        axis.plot(
            100 * group.coverage_fraction,
            group.probe_backup_distance_mean,
            "o-",
            lw=1.8,
            ms=4,
            color=colors[name],
            label=labels[name],
        )
    axis.set_xscale("log")
    axis.set_xticks([1, 5, 10, 20, 50, 100], labels=["1", "5", "10", "20", "50", "100"])
    axis.set_xlabel("Pairs retained as nearest by each distance (%)")
    axis.set_ylabel("Independent nonlinear-probe backup distance ↓")
    axis.set_title("Distance-based pair selection")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.text(
        0.5,
        0.035,
        "Same-action cross-user pairs · 4,096 users · 8,192 contexts · 32 repeats · 8 probes",
        ha="center",
        fontsize=8,
    )
    fig.savefig(args.output_dir / "distance_ranking_diagnostic.png", dpi=220)
    fig.savefig(args.output_dir / "distance_ranking_diagnostic.pdf")
    plt.close(fig)

    at_one = frame[frame.coverage_fraction == 0.01].copy()
    result = at_one[["distance", "pairs", "users", "probe_backup_distance_mean", "probe_standard_error_mean"]]
    result.to_csv(args.output_dir / "distance_ranking_at_1pct.csv", index=False)
    brief = """# Existing M2 cache: distance-ranking diagnostic

This reuses the predeclared M2 cache; it does not rerun the environment and is **not** an actual random/activity/value/HAVL grouping comparison. All 200,000 candidate pairs are cross-user, use the exact same action, and are evaluated by independent nonlinear continuation probes (4 frozen trained-Q probes and 4 random probes, 32 environment repeats).

At 1% retained coverage, mean independent backup distance is 0.355 for current value, 0.251 for critic embedding, 0.223 for raw state, and 0.254 for the HAVL descriptor. Thus the descriptor is substantially more informative than scalar value, but it is not uniformly best: raw-state and critic-embedding rankings are better in this cache. The defensible takeaway is that scalar value alone is insufficient and backup compatibility contains additional structure—not that the current HAVL grouping dominates every representation.
"""
    (args.output_dir / "DISTANCE_RANKING_BRIEF_ZH.md").write_text(brief, encoding="utf-8")


if __name__ == "__main__":
    main()
