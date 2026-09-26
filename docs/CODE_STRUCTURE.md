# Code Structure

## HAVL implementation

- `src/bellman_sharing/profile.py`: auxiliary reward/continuation/successor
  prediction, Bellman-profile distance, prototype fitting, and soft membership.
- `src/bellman_sharing/controller.py`: extraction of the optimizer's actual
  proposal, group-gradient constraints, minimum-change projection, and nonlinear
  acceptance/backtracking.
- `src/bellman_sharing/main_table.py`: online fitted-DQN training loop that joins
  HAVL grouping and transfer-constrained value updates; also contains declared
  comparison variants.
- `src/bellman_sharing/kuaisim_adapter.py`: single-item interface around native
  KuaiSim, including replay collection and fixed-context rollouts.
- `src/bellman_sharing/models.py`: critic, profile predictor, and auxiliary models.
- `src/bellman_sharing/data.py` and `training.py`: replay structures, profile
  pretraining, TD targets, evaluation, and shared training helpers.
- `src/bellman_sharing/experiments/`: dataset-partition and checkpoint-snapshot
  utilities used by the current HAVL experiments.

## Entrypoints

- `scripts/run_main_table_dqn.py`: configured Base/HAVL/alternative-grouping run.
- `scripts/aggregate_main_table.py`: aggregates completed main-table runs.
- `scripts/run_section54_update_audit.py`: paired one-step harm/benefit audit.
- `scripts/summarize_section54_update_audit.py`: paired conditional intervals and
  CSV/brief generation.
- `scripts/plot_section54_distance_ranking.py`: cached distance-ranking diagnostic;
  it is not an actual grouping comparison.

## KuaiSim subset

`third_party/kuaisim/` preserves only the modules imported by the HAVL adapter:

- `code/env/`: base environment and whole-session GPU environment.
- `code/reader/`: KuaiRand sequence readers.
- `code/model/simulator/`: user-response simulator classes.
- `code/model/*.py` and `code/utils.py`: shared model and utility functions.
- `scripts/`: KuaiRand-27K download and feature-preparation helpers.

The upstream repository was captured at commit
`a1ff37cdf528cba539eab993a4a0819d8da05847`. Dataset files, notebooks, native RL
agents not used by HAVL, pretrained output, and caches are omitted.
