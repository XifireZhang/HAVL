#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/main_table_v1"
FORMAL_SEEDS = (0, 1, 2)
DATASETS = ("kuairand_1k", "kuairand_27k_supported_subset")
BACKBONES = ("dqn", "a2c", "ddpg", "td3")
METHODS = (
    "base", "user_uniform", "rlur_adapted", "auro_adapted", "gem_adapted",
    "discor_adapted", "havl_random", "havl_activity", "havl_value", "havl",
)


def load_rows() -> list[dict[str, str]]:
    path = REPORT / "seed_results.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["status"] == "complete"]


def cell(rows: list[dict[str, str]], dataset: str, backbone: str, method: str) -> dict:
    selected = {
        int(row["seed"]): float(row["avg_total_reward"])
        for row in rows
        if row["dataset"] == dataset and row["backbone"] == backbone
        and row["method"] == method and int(row["seed"]) in FORMAL_SEEDS
    }
    values = [selected[seed] for seed in FORMAL_SEEDS if seed in selected]
    if len(values) != len(FORMAL_SEEDS):
        return {"status": f"pending {len(values)}/{len(FORMAL_SEEDS)}", "display": "pending"}
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    return {
        "status": "complete", "mean": mean, "seed_sd": sd,
        "seeds": list(FORMAL_SEEDS), "display": f"{mean:.4f} ± {sd:.4f}",
    }


def main() -> None:
    rows = load_rows()
    columns = [(dataset, backbone) for dataset in DATASETS for backbone in BACKBONES]
    output = []
    for method in METHODS:
        record: dict[str, str] = {"method": method}
        for dataset, backbone in columns:
            record[f"{dataset}/{backbone}"] = cell(rows, dataset, backbone, method)["display"]
        output.append(record)
    csv_path = REPORT / "main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    latex_lines = [
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"{dataset}/{backbone.upper()}" for dataset, backbone in columns) + r" \\",
        r"\midrule",
    ]
    for record in output:
        latex_lines.append(
            record["method"].replace("_", r"\_") + " & "
            + " & ".join(record[f"{dataset}/{backbone}"] for dataset, backbone in columns)
            + r" \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (REPORT / "main_table.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    completed = sum(value != "pending" for record in output for key, value in record.items() if key != "method")
    markdown = [
        "# Main Table Results",
        "",
        f"Updated {datetime.now(timezone.utc).isoformat()}.",
        "",
        f"Completed cells: {completed}/{len(METHODS) * len(columns)}. Formal aggregation uses only predeclared seeds 0, 1, and 2.",
        "",
        "Values are mean ± sample standard deviation across independent training seeds. Pending cells are never filled with zero or development results.",
        "",
    ]
    (REPORT / "MAIN_TABLE_RESULTS.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
