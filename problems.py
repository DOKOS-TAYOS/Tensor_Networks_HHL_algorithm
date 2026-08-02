"""Public problem-builder compatibility functions."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from experiments.problem_builders import (
    build_damped_oscillator,
    build_harmonic_oscillator,
    build_heat_problem,
)


def _oscillator_params(param: Mapping[str, float]) -> dict[str, float]:
    dt = float(param["dt"])
    steps = int(param["steps"])
    return {
        "T": steps * dt,
        "dt": dt,
        "k": float(param["k"]),
        "m": float(param["m"]),
        "nu": float(param["nu"]),
        "C": float(param["C"]),
        "x0": float(param["x0"]),
        "xT": float(param["xq"]),
    }


def problem_OAF(
    param: Mapping[str, float], scaling: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the corrected forced harmonic oscillator interior system.

    ``steps`` retains its historical meaning of time intervals. Thus
    ``steps=100`` gives 101 nodes, two prescribed endpoints, and 99 unknowns.
    """
    problem = build_harmonic_oscillator(_oscillator_params(param), scale=scaling)
    return (
        problem["rhs"].to(dtype=torch.complex128),
        problem["matrix"].to(dtype=torch.complex128),
    )


def problem_OAA(
    param: Mapping[str, float], scaling: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the corrected damped system and its Hermitian embedding."""
    parameters = _oscillator_params(param)
    parameters["gamma"] = float(param["gamma"])
    problem = build_damped_oscillator(parameters, scale=scaling)
    return (
        problem["embedded_rhs"].to(dtype=torch.complex128),
        problem["embedding"].to(dtype=torch.complex128),
        problem["rhs"].to(dtype=torch.complex128),
        problem["matrix"].to(dtype=torch.complex128),
    )


def problem_C2D(
    param: Mapping[str, float], scaling: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the existing two-dimensional static heat-equation system."""
    problem = build_heat_problem(param, scale=scaling)
    return (
        problem["rhs"].to(dtype=torch.complex128),
        problem["matrix"].to(dtype=torch.complex128),
    )
