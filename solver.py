"""Convenience solver for the example application."""

from __future__ import annotations

import numpy as np
import torch

from experiments.problem_builders import reconstruct_with_boundaries
from problems import problem_C2D, problem_OAA, problem_OAF
from tn_hhl import tensornetwork_HHL

SolverVector = np.ndarray | torch.Tensor


def solve_problem(
    problem: str, params: dict[str, float], num_eigen: int, t: float
) -> tuple[SolverVector, SolverVector, np.ndarray, object | None]:
    """Solve an OAF, OAA, or C2D problem with TN HHL and direct solve."""
    if problem == "OAF":
        force, matrix = problem_OAF(params, scaling=True)
        interior_tn = tensornetwork_HHL(num_eigen, t, force, matrix)
        interior_reference = torch.linalg.solve(matrix.real, force.real)
        algorithm_result = reconstruct_with_boundaries(
            interior_tn, params["x0"], params["xq"]
        )
        actual_result = reconstruct_with_boundaries(
            interior_reference, params["x0"], params["xq"]
        )
        x_axis = np.arange(int(params["steps"]) + 1, dtype=float) * params["dt"]
        return algorithm_result, actual_result, x_axis, None

    if problem == "OAA":
        force, matrix, physical_force, physical_matrix = problem_OAA(
            params, scaling=True
        )
        embedded_tn = tensornetwork_HHL(num_eigen, t, force, matrix)
        n_interior = int(params["steps"]) - 1
        interior_tn = embedded_tn[n_interior:]
        interior_reference = torch.linalg.solve(
            physical_matrix.real, physical_force.real
        )
        algorithm_result = reconstruct_with_boundaries(
            interior_tn, params["x0"], params["xq"]
        )
        actual_result = reconstruct_with_boundaries(
            interior_reference, params["x0"], params["xq"]
        )
        x_axis = np.arange(int(params["steps"]) + 1, dtype=float) * params["dt"]
        return algorithm_result, actual_result, x_axis, None

    if problem == "C2D":
        force, matrix = problem_C2D(params, scaling=False)
        algorithm_result = tensornetwork_HHL(num_eigen, t, force, matrix)
        actual_result = torch.linalg.solve(matrix.real, force.real)
        x_axis = (
            np.arange(int(params["nx"] * params["ny"]), dtype=float) * params["dxy"]
        )
        result_2d: list[list[object]] = list(
            algorithm_result.reshape(int(params["nx"]), int(params["ny"]))
        )
        result_2d = [
            [params["u1x"]] * (int(params["nx"]) + 2),
            *result_2d,
            [params["u2x"]] * (int(params["nx"]) + 2),
        ]
        for index in range(1, int(params["nx"]) + 1):
            result_2d[index] = [
                params["u1y"],
                *result_2d[index],
                params["u2y"],
            ]
        return algorithm_result, actual_result, x_axis, result_2d

    raise ValueError(f"Unknown problem type: {problem}")
