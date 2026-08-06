from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.parameter_selection import (
    choose_parameter_candidate,
    hhl_filter,
    search_parameters,
)
from experiments.problem_builders import (
    build_damped_oscillator,
    build_harmonic_oscillator,
    build_heat_problem,
    generate_random_problems,
    oscillator_grid,
)
from experiments.run_reviewer_r1_c5_c6 import (
    log_scale_confidence_errors,
    normalized_residual,
    relative_solution_error,
    rmse,
    spectral_diagnostics,
)

CONFIG_PATH = Path("experiments/config_r1_c5_c6.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _direct_filter(eigenvalues: np.ndarray, mu: int, tau: float) -> np.ndarray:
    bins = np.arange(mu)
    signed_bins = np.where(bins <= mu // 2, bins, bins - mu)
    weights = np.zeros(mu)
    weights[signed_bins != 0] = 1.0 / signed_bins[signed_bins != 0]
    values = []
    for eigenvalue in eigenvalues:
        kernels = [
            sum(np.exp(2j * np.pi * b * (tau * eigenvalue - d) / mu) for b in range(mu))
            for d in bins
        ]
        values.append(tau / mu**2 * np.sum(weights * np.abs(kernels) ** 2))
    return np.asarray(values)


@pytest.mark.parametrize("mu", [3, 4, 7, 8])
def test_hhl_filter_matches_independent_direct_sum(mu: int) -> None:
    eigenvalues = np.array([-1.31, 0.73, 2.11])
    assert hhl_filter(eigenvalues, mu, 1.17) == pytest.approx(
        _direct_filter(eigenvalues, mu, 1.17), abs=2e-12
    )


def test_filter_reproduces_inverse_on_an_exact_nonaliased_grid() -> None:
    eigenvalues = np.array([-2.0, 1.0])
    assert hhl_filter(eigenvalues, mu=16, tau=2.0) == pytest.approx(
        1.0 / eigenvalues, abs=1e-12
    )


def test_selector_uses_cost_then_falls_back_to_filter_error() -> None:
    def candidate(cost: int, error: float, target_met: bool) -> dict[str, object]:
        return {
            "feasible": True,
            "cost_proxy": cost,
            "rhs_relative_filter_error": error,
            "mu": cost,
            "tau": 1.0,
            "target_met": target_met,
            "selected": False,
        }

    accurate = candidate(200, 0.001, True)
    acceptable = candidate(100, 0.009, True)
    selected, _ = choose_parameter_candidate([accurate, acceptable])
    assert selected["cost_proxy"] == 100

    lower_error = candidate(200, 0.02, False)
    lower_cost = candidate(100, 0.03, False)
    selected, _ = choose_parameter_candidate([lower_cost, lower_error])
    assert selected["rhs_relative_filter_error"] == pytest.approx(0.02)


def test_default_damped_selector_meets_the_stated_target() -> None:
    problem = build_damped_oscillator(CONFIG["problems"]["damped_oscillator"])
    selected = search_parameters(
        problem["embedding"], problem["embedded_rhs"], problem["name"], CONFIG
    )["selected"]

    assert selected["mu"] == 4096
    assert selected["target_met"]
    assert selected["rhs_relative_filter_error"] <= 0.01


def test_application_discretizations_have_the_paper_dimensions() -> None:
    grid = oscillator_grid(T=50.0, dt=0.5)
    harmonic = build_harmonic_oscillator(CONFIG["problems"]["harmonic_oscillator"])
    damped = build_damped_oscillator(CONFIG["problems"]["damped_oscillator"])
    heat = build_heat_problem(CONFIG["problems"]["heat_2d"])

    assert (grid["n_intervals"], grid["n_nodes"], grid["n_interior"]) == (
        100,
        101,
        99,
    )
    assert harmonic["matrix"].shape == (99, 99)
    assert damped["matrix"].shape == (99, 99)
    assert damped["embedding"].shape == (198, 198)
    assert heat["matrix"].shape == (400, 400)
    assert torch.allclose(damped["embedding"], damped["embedding"].T)
    assert torch.allclose(heat["matrix"], heat["matrix"].T)


def test_oscillator_force_amplitudes_use_the_explicit_config_key() -> None:
    harmonic_params = CONFIG["problems"]["harmonic_oscillator"]
    damped_params = CONFIG["problems"]["damped_oscillator"]

    assert harmonic_params["force_amplitude"] == 9.0
    assert damped_params["force_amplitude"] == 9.0
    assert "C" not in harmonic_params
    assert "C" not in damped_params


def test_harmonic_oscillator_system_is_numerically_unchanged() -> None:
    problem = build_harmonic_oscillator(
        CONFIG["problems"]["harmonic_oscillator"], scale=False
    )
    expected_diagonal = -2.0 + 5.0 / 7.0 * 0.5**2
    expected_rhs = 0.5**2 * 9.0 * np.sin(np.pi * 0.4 * np.arange(0.5, 50.0, 0.5))
    expected_rhs[0] -= 5.0
    expected_rhs[-1] -= 3.0

    assert problem["matrix"].numpy() == pytest.approx(
        np.diag(np.full(99, expected_diagonal))
        + np.diag(np.ones(98), k=1)
        + np.diag(np.ones(98), k=-1)
    )
    assert problem["rhs"].numpy() == pytest.approx(expected_rhs)


def test_random_systems_are_deterministic_symmetric_and_sparse() -> None:
    random_config = {**CONFIG["random"], "singular_tolerance": 1e-12}
    first = generate_random_problems(random_config)
    second = generate_random_problems(random_config)
    expected_upper_entries = round(0.25 * (16 * 15 // 2))

    assert len(first) == 20
    for first_problem, second_problem in zip(first, second, strict=True):
        matrix = first_problem["matrix"].numpy()
        assert matrix == pytest.approx(second_problem["matrix"].numpy())
        assert first_problem["rhs"].numpy() == pytest.approx(
            second_problem["rhs"].numpy()
        )
        assert first_problem["seed"] == 12345
        assert np.count_nonzero(np.triu(matrix, k=1)) == expected_upper_entries
        assert matrix == pytest.approx(matrix.T)
        assert np.max(np.abs(np.linalg.eigvalsh(matrix))) == pytest.approx(1.0 / 1.01)


def test_metrics_and_spectral_diagnostics_match_hand_calculations() -> None:
    estimate = np.array([2.0, 0.0])
    reference = np.array([1.0, 1.0])

    assert rmse(estimate, reference) == pytest.approx(1.0)
    assert relative_solution_error(estimate, reference) == pytest.approx(1.0)
    assert normalized_residual(np.eye(2), estimate, reference) == pytest.approx(1.0)

    diagnostics, eigenvalues = spectral_diagnostics(
        np.diag([-2.0, 1.0]), tau=2.0, mu=16
    )
    assert eigenvalues == pytest.approx([-2.0, 1.0])
    assert diagnostics["condition_number"] == pytest.approx(2.0)
    assert diagnostics["max_grid_distance"] == pytest.approx(0.0)
    assert diagnostics["no_aliasing"]


def test_log_confidence_errors_omit_nonpositive_lower_arms() -> None:
    errors, omitted = log_scale_confidence_errors(
        means=np.array([1.0, 2.0, 3.0]),
        lows=np.array([0.5, 0.0, -1.0]),
        highs=np.array([1.5, 3.0, 5.0]),
    )

    np.testing.assert_allclose(errors, [[0.5, 0.0, 0.0], [0.5, 1.0, 2.0]])
    assert omitted.tolist() == [False, True, True]
