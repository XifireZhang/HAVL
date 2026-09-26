from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable

import torch


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def flatten_tensors(values: Iterable[torch.Tensor]) -> torch.Tensor:
    blocks = [value.reshape(-1) for value in values]
    return torch.cat(blocks) if blocks else torch.empty(0)


def assign_flat(parameters: list[torch.nn.Parameter], value: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            parameter.copy_(value[offset:offset + size].view_as(parameter))
            offset += size
    if offset != value.numel():
        raise ValueError("flat vector does not match parameters")


def actual_optimizer_candidate(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance optimizer state once, but restore parameters and return actual displacement."""
    parameters = trainable_parameters(module)
    original = flatten_tensors(parameter.detach().clone() for parameter in parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    candidate = flatten_tensors(parameter.detach().clone() for parameter in parameters)
    assign_flat(parameters, original)
    return original, candidate - original


def flat_gradient(loss: torch.Tensor, module: torch.nn.Module, retain_graph: bool) -> torch.Tensor:
    parameters = trainable_parameters(module)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return flatten_tensors(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ).detach()


@dataclass(frozen=True)
class ProjectionResult:
    displacement: torch.Tensor
    multipliers: torch.Tensor
    trust_multiplier: float
    maximum_linear_violation: float
    norm_violation: float
    iterations: int
    converged: bool


def _halfspace_solution(
    d0: torch.Tensor,
    gradients: torch.Tensor,
    budgets: torch.Tensor,
    trust_multiplier: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    if gradients.numel() == 0:
        return d0 / (1.0 + trust_multiplier), torch.zeros(0, device=d0.device), 0, True
    gram = gradients @ gradients.T
    multipliers = torch.zeros(len(gradients), dtype=d0.dtype, device=d0.device)
    denominator_scale = 1.0 + trust_multiplier
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = multipliers.clone()
        displacement = (d0 - gradients.T @ multipliers) / denominator_scale
        for group in range(len(gradients)):
            diagonal = gram[group, group]
            if diagonal <= 1e-20:
                continue
            violation = torch.dot(gradients[group], displacement) - budgets[group]
            updated = torch.clamp(
                multipliers[group] + violation * denominator_scale / diagonal,
                min=0.0,
            )
            displacement = displacement - gradients[group] * (
                updated - multipliers[group]
            ) / denominator_scale
            multipliers[group] = updated
        if torch.max(torch.abs(multipliers - previous)) <= tolerance:
            converged = True
            break
    return displacement, multipliers, iteration, converged


def project_candidate(
    d0: torch.Tensor,
    gradients: torch.Tensor,
    budgets: torch.Tensor,
    radius: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> ProjectionResult:
    """Euclidean projection using the small group-gradient Gram system and KKT search."""
    d0 = d0.detach()
    gradients = gradients.detach()
    budgets = budgets.detach()
    if radius < 0 or torch.any(budgets < 0):
        raise ValueError("radius and budgets must be nonnegative")
    if gradients.ndim != 2 or gradients.shape[1] != d0.numel() or budgets.shape != (len(gradients),):
        raise ValueError("inconsistent projection shapes")
    if radius == 0:
        zero = torch.zeros_like(d0)
        return ProjectionResult(zero, torch.zeros(len(gradients), device=d0.device), 0.0, 0.0, 0.0, 0, True)
    displacement, multipliers, iterations, converged = _halfspace_solution(
        d0, gradients, budgets, 0.0, tolerance, max_iterations
    )
    trust_multiplier = 0.0
    if torch.linalg.vector_norm(displacement) > radius + tolerance:
        low, high = 0.0, 1.0
        for _ in range(80):
            trial, _, _, _ = _halfspace_solution(
                d0, gradients, budgets, high, tolerance, max_iterations
            )
            if torch.linalg.vector_norm(trial) <= radius:
                break
            high *= 2.0
        for _ in range(60):
            middle = (low + high) / 2.0
            trial, trial_multipliers, trial_iterations, trial_converged = _halfspace_solution(
                d0, gradients, budgets, middle, tolerance, max_iterations
            )
            if torch.linalg.vector_norm(trial) > radius:
                low = middle
            else:
                high = middle
                displacement, multipliers = trial, trial_multipliers
                iterations += trial_iterations
                converged = converged and trial_converged
        trust_multiplier = high
    linear_violation = (
        float(torch.clamp(gradients @ displacement - budgets, min=0).max())
        if len(gradients) else 0.0
    )
    norm_violation = max(0.0, float(torch.linalg.vector_norm(displacement)) - radius)
    return ProjectionResult(
        displacement=displacement,
        multipliers=multipliers,
        trust_multiplier=trust_multiplier,
        maximum_linear_violation=linear_violation,
        norm_violation=norm_violation,
        iterations=iterations,
        converged=converged and linear_violation <= 10 * tolerance and norm_violation <= 10 * tolerance,
    )


@dataclass(frozen=True)
class BacktrackingResult:
    displacement: torch.Tensor
    scale: float
    losses_before: torch.Tensor
    losses_after: torch.Tensor
    accepted: bool


def apply_with_backtracking(
    module: torch.nn.Module,
    original: torch.Tensor,
    displacement: torch.Tensor,
    loss_closure: Callable[[], torch.Tensor],
    tolerance: float,
    max_steps: int,
) -> BacktrackingResult:
    parameters = trainable_parameters(module)
    assign_flat(parameters, original)
    with torch.no_grad():
        before = loss_closure().detach().clone()
    for step in range(max_steps + 1):
        scale = 0.5 ** step
        assign_flat(parameters, original + scale * displacement)
        with torch.no_grad():
            after = loss_closure().detach().clone()
        if torch.all(after <= before + tolerance):
            return BacktrackingResult(scale * displacement, scale, before, after, True)
    assign_flat(parameters, original)
    return BacktrackingResult(torch.zeros_like(displacement), 0.0, before, before, False)


def apply_with_loss_budgets(
    module: torch.nn.Module,
    original: torch.Tensor,
    displacement: torch.Tensor,
    loss_closure: Callable[[], torch.Tensor],
    allowed_increase: torch.Tensor,
    numerical_tolerance: float,
    max_steps: int,
) -> BacktrackingResult:
    """Backtrack against explicit research budgets plus a numerical tolerance.

    ``allowed_increase`` changes the algorithmic constraint, while
    ``numerical_tolerance`` only absorbs floating-point error. Keeping the two
    separate prevents a permissive tolerance from silently defining a new
    controller.
    """
    parameters = trainable_parameters(module)
    assign_flat(parameters, original)
    with torch.no_grad():
        before = loss_closure().detach().clone()
    budget = allowed_increase.detach().to(device=before.device, dtype=before.dtype)
    if budget.shape != before.shape or torch.any(budget < 0):
        raise ValueError("loss budgets must be nonnegative and match the losses")
    for step in range(max_steps + 1):
        scale = 0.5 ** step
        assign_flat(parameters, original + scale * displacement)
        with torch.no_grad():
            after = loss_closure().detach().clone()
        if torch.all(after <= before + budget + numerical_tolerance):
            return BacktrackingResult(scale * displacement, scale, before, after, True)
    assign_flat(parameters, original)
    return BacktrackingResult(torch.zeros_like(displacement), 0.0, before, before, False)
