# HAVL Research Code

This repository contains the code required to develop and audit
**Heterogeneity-Aware Value Learning (HAVL)** on a KuaiSim-based recommendation
environment. It is a source-only snapshot: datasets, pretrained KuaiSim weights,
training checkpoints, replay buffers, and experiment outputs are deliberately
excluded.

## What is included

- `src/bellman_sharing/`: HAVL descriptors, soft grouping, transfer-constrained
  critic updates, DQN training, models, data helpers, and diagnostic utilities.
- `scripts/`: the configured DQN experiment and Section 5.4 mechanism audit.
- `configs/`: the current KuaiRand-27K/HAVL experiment configurations.
- `tests/`: focused tests for grouping, controller, replay handling, and DQN code.
- `third_party/kuaisim/`: the minimal KuaiSim environment/reader/simulator source
  used by `NativeKuaiSimAdapter`, plus the upstream license and data-preparation
  scripts.

See [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) for the detailed mapping.

## External files required to run experiments

The configured experiment expects these files, which are not committed:

```text
third_party/kuaisim/dataset/kuairand/Kuairand-27K/data/
third_party/kuaisim/code/output/Kuairand_27K/env/
```

The first directory contains KuaiRand-27K CSV files. The second contains the
pretrained KuaiSim response-model log and checkpoint. Update the `[kuaisim]`
paths in the TOML configuration if these files are stored elsewhere.

## Environment and commands

Install a CUDA-compatible PyTorch build first, then install this repository:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the configured DQN comparison from the repository root:

```bash
PYTHONPATH=src python scripts/run_main_table_dqn.py \
  --config configs/main_table_v1/dqn_27k_dev_seed1001.toml
```

The Section 5.4 audit consumes previously generated replay/profile/checkpoint
artifacts. Place them under `artifacts/` according to
[artifacts/README.md](artifacts/README.md), then run:

```bash
PYTHONPATH=src python scripts/run_section54_update_audit.py \
  --config configs/section54/update_audit_mid.toml
```

## Reproducibility boundary

This commit preserves code and configuration, not the large experimental state.
It cannot reproduce numerical results until the omitted KuaiSim model, dataset,
and declared checkpoint artifacts are restored. No credentials or server-specific
SSH settings are included.

