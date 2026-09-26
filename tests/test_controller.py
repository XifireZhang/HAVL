from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from bellman_sharing.controller import (
    actual_optimizer_candidate,
    apply_with_backtracking,
    apply_with_loss_budgets,
    assign_flat,
    flatten_tensors,
    project_candidate,
    trainable_parameters,
)
from bellman_sharing.models import QCritic


class ControllerTests(unittest.TestCase):
    def test_feasible_candidate_is_unchanged(self) -> None:
        d0 = torch.tensor([.2, -.1], dtype=torch.float64)
        gradients = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        result = project_candidate(d0, gradients, torch.tensor([.3], dtype=torch.float64), 1.0)
        torch.testing.assert_close(result.displacement, d0, atol=1e-8, rtol=1e-8)
        self.assertTrue(result.converged)

    def test_conflicting_constraints_have_zero_safe_solution(self) -> None:
        d0 = torch.tensor([1.0], dtype=torch.float64)
        gradients = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        result = project_candidate(d0, gradients, torch.zeros(2, dtype=torch.float64), 2.0)
        self.assertLessEqual(abs(float(result.displacement)), 1e-7)
        self.assertLessEqual(result.maximum_linear_violation, 1e-7)

    def test_trust_radius_is_enforced(self) -> None:
        result = project_candidate(
            torch.tensor([3.0, 4.0], dtype=torch.float64),
            torch.empty((0, 2), dtype=torch.float64), torch.empty(0, dtype=torch.float64), 1.0,
        )
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result.displacement)), 1.0, places=6)

    def test_zero_gradient_and_zero_radius_fallback(self) -> None:
        result = project_candidate(
            torch.tensor([1.0]), torch.zeros((1, 1)), torch.zeros(1), 0.0
        )
        torch.testing.assert_close(result.displacement, torch.zeros(1))

    def test_finite_difference_matches_linear_influence_sign(self) -> None:
        parameter = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([1.0, .5], dtype=torch.float64)
        loss = .5 * ((parameter - target) ** 2).sum()
        gradient = torch.autograd.grad(loss, parameter)[0]
        delta = torch.tensor([.01, -.02], dtype=torch.float64)
        epsilon = 1e-5
        def objective(value: torch.Tensor) -> torch.Tensor:
            return .5 * ((value - target) ** 2).sum()
        finite = (objective(parameter + epsilon * delta) - objective(parameter)) / epsilon
        self.assertAlmostEqual(float(finite), float(torch.dot(gradient, delta)), places=5)

    def test_optimizer_candidate_is_actual_adam_displacement_and_restores_parameters(self) -> None:
        torch.manual_seed(2)
        model = torch.nn.Linear(2, 1, bias=False)
        twin = deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        twin_optimizer = torch.optim.Adam(twin.parameters(), lr=.01)
        x = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[.3]])
        original = flatten_tensors(p.detach().clone() for p in model.parameters())
        loss = ((model(x) - target) ** 2).mean()
        returned_original, displacement = actual_optimizer_candidate(model, optimizer, loss)
        torch.testing.assert_close(returned_original, original)
        torch.testing.assert_close(flatten_tensors(model.parameters()), original)
        twin_optimizer.zero_grad(); ((twin(x) - target) ** 2).mean().backward(); twin_optimizer.step()
        expected = flatten_tensors(twin.parameters()) - original
        torch.testing.assert_close(displacement, expected)

    def test_backtracking_checks_actual_nonlinear_loss_and_can_fallback(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        original = flatten_tensors(model.parameters()).detach().clone()
        displacement = torch.tensor([2.0])
        def closure() -> torch.Tensor:
            return (model.weight.reshape(-1) ** 2)
        result = apply_with_backtracking(model, original, displacement, closure, 0.0, 3)
        self.assertFalse(result.accepted)
        self.assertEqual(result.scale, 0.0)
        torch.testing.assert_close(flatten_tensors(model.parameters()), original)

    def test_nonlinear_critic_control_covers_all_trainable_layers(self) -> None:
        critic = QCritic(3, 2, [8, 4])
        flat = flatten_tensors(trainable_parameters(critic))
        self.assertEqual(flat.numel(), sum(parameter.numel() for parameter in critic.parameters()))

    def test_budgeted_backtracking_preserves_feasible_proposal(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        original = flatten_tensors(model.parameters()).detach().clone()
        displacement = torch.tensor([0.1])

        def closure() -> torch.Tensor:
            return model.weight.reshape(-1).square()

        result = apply_with_loss_budgets(
            model,
            original,
            displacement,
            closure,
            allowed_increase=torch.tensor([0.02]),
            numerical_tolerance=1e-8,
            max_steps=4,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.scale, 1.0)
        torch.testing.assert_close(result.displacement, displacement)

    def test_budget_is_not_confused_with_numerical_tolerance(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        original = flatten_tensors(model.parameters()).detach().clone()

        def closure() -> torch.Tensor:
            return model.weight.reshape(-1).square()

        result = apply_with_loss_budgets(
            model,
            original,
            torch.tensor([1.0]),
            closure,
            allowed_increase=torch.tensor([0.2]),
            numerical_tolerance=1e-9,
            max_steps=3,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.scale, 0.25)


if __name__ == "__main__":
    unittest.main()
