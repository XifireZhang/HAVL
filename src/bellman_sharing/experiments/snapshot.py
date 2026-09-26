from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch


@dataclass
class LearnerSnapshot:
    modules: dict[str, dict[str, torch.Tensor]]
    optimizers: dict[str, dict[str, Any]]
    python_rng: object
    numpy_rng: tuple[Any, ...]
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None
    extras: dict[str, Any]


def capture_snapshot(
    modules: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    extras: dict[str, Any] | None = None,
) -> LearnerSnapshot:
    return LearnerSnapshot(
        modules={name: deepcopy(module.state_dict()) for name, module in modules.items()},
        optimizers={
            name: deepcopy(optimizer.state_dict()) for name, optimizer in optimizers.items()
        },
        python_rng=random.getstate(),
        numpy_rng=deepcopy(np.random.get_state()),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
        extras=deepcopy(extras or {}),
    )


def restore_snapshot(
    snapshot: LearnerSnapshot,
    modules: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, Any]:
    if set(modules) != set(snapshot.modules):
        raise ValueError("module names do not match snapshot")
    if set(optimizers) != set(snapshot.optimizers):
        raise ValueError("optimizer names do not match snapshot")
    for name, module in modules.items():
        module.load_state_dict(deepcopy(snapshot.modules[name]))
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(deepcopy(snapshot.optimizers[name]))
    random.setstate(snapshot.python_rng)
    np.random.set_state(deepcopy(snapshot.numpy_rng))
    torch.set_rng_state(snapshot.torch_rng.clone())
    if snapshot.cuda_rng is not None:
        torch.cuda.set_rng_state_all([state.clone() for state in snapshot.cuda_rng])
    return deepcopy(snapshot.extras)

