"""Matrices and right-hand sides used in the reviewer experiments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def oscillator_grid(T: float = 50.0, dt: float = 0.5) -> dict[str, object]:
    """Return the nodes and dimensions of the oscillator boundary problem."""
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    n_intervals = round(T / dt)
    if not np.isclose(n_intervals * dt, T, atol=1e-12, rtol=0.0):
        raise ValueError("T must be an integer multiple of dt")

    n_nodes = n_intervals + 1
    return {
        "T": float(T),
        "dt": float(dt),
        "n_intervals": n_intervals,
        "n_nodes": n_nodes,
        "n_interior": n_nodes - 2,
        "t_nodes": np.arange(n_nodes, dtype=float) * dt,
    }


def _scale_system(
    matrix: torch.Tensor, rhs: torch.Tensor, scale: bool
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if not scale:
        return matrix, rhs, 1.0
    factor = float(torch.linalg.norm(matrix).item())
    if factor == 0.0:
        raise ValueError("cannot scale a zero matrix")
    return matrix / factor, rhs / factor, factor


def build_harmonic_oscillator(
    params: Mapping[str, float], *, scale: bool = True
) -> dict[str, object]:
    """Build the 99-unknown forced harmonic oscillator system."""
    grid = oscillator_grid(T=float(params["T"]), dt=float(params["dt"]))
    k, mass = float(params["k"]), float(params["m"])
    frequency, amplitude = float(params["nu"]), float(params["C"])
    x0, xT = float(params["x0"]), float(params["xT"])
    dt = float(grid["dt"])
    n = int(grid["n_interior"])

    diagonal = -2.0 + k / mass * dt**2
    matrix = torch.diag(torch.full((n,), diagonal, dtype=torch.float64))
    off_diagonal = torch.ones(n - 1, dtype=torch.float64)
    matrix += torch.diag(off_diagonal, diagonal=1)
    matrix += torch.diag(off_diagonal, diagonal=-1)

    times = np.asarray(grid["t_nodes"])[1:-1]
    rhs_values = dt**2 * amplitude * np.sin(np.pi * frequency * times)
    rhs_values[0] -= x0
    rhs_values[-1] -= xT
    rhs = torch.as_tensor(rhs_values, dtype=torch.float64)
    matrix, rhs, scale_factor = _scale_system(matrix, rhs, scale)

    return {
        "name": "harmonic_oscillator",
        "matrix": matrix,
        "rhs": rhs,
        "matrix_role": "physical_hermitian_99x99",
        "scale_factor": scale_factor,
        "grid": grid,
        "x0": x0,
        "xT": xT,
    }


def build_damped_oscillator(
    params: Mapping[str, float], *, scale: bool = True
) -> dict[str, object]:
    """Build the damped 99-square system and its 198-square embedding."""
    grid = oscillator_grid(T=float(params["T"]), dt=float(params["dt"]))
    k, mass = float(params["k"]), float(params["m"])
    frequency, amplitude = float(params["nu"]), float(params["C"])
    x0, xT = float(params["x0"]), float(params["xT"])
    dt = float(grid["dt"])
    n = int(grid["n_interior"])
    gamma = float(params["gamma"])
    beta_minus = 1.0 - gamma * dt / 2.0
    beta_plus = 1.0 + gamma * dt / 2.0

    diagonal = -2.0 + k / mass * dt**2
    matrix = torch.diag(torch.full((n,), diagonal, dtype=torch.float64))
    matrix += torch.diag(
        torch.full((n - 1,), beta_plus, dtype=torch.float64), diagonal=1
    )
    matrix += torch.diag(
        torch.full((n - 1,), beta_minus, dtype=torch.float64), diagonal=-1
    )

    times = np.asarray(grid["t_nodes"])[1:-1]
    rhs_values = dt**2 * amplitude * np.sin(np.pi * frequency * times)
    rhs_values[0] -= beta_minus * x0
    rhs_values[-1] -= beta_plus * xT
    rhs = torch.as_tensor(rhs_values, dtype=torch.float64)
    matrix, rhs, scale_factor = _scale_system(matrix, rhs, scale)

    embedding = torch.zeros((2 * n, 2 * n), dtype=torch.float64)
    embedding[:n, n:] = matrix
    embedding[n:, :n] = matrix.T
    embedded_rhs = torch.zeros(2 * n, dtype=torch.float64)
    embedded_rhs[:n] = rhs

    return {
        "name": "damped_oscillator",
        "matrix": matrix,
        "rhs": rhs,
        "embedding": embedding,
        "embedded_rhs": embedded_rhs,
        "matrix_role": "hermitian_embedding_198x198",
        "scale_factor": scale_factor,
        "grid": grid,
        "x0": x0,
        "xT": xT,
        "beta_minus": beta_minus,
        "beta_plus": beta_plus,
    }


def build_heat_problem(
    params: Mapping[str, float], *, scale: bool = False
) -> dict[str, object]:
    """Build the existing two-dimensional static heat system."""
    k = float(params["k"])
    u1x, u2x = float(params["u1x"]), float(params["u2x"])
    u1y, u2y = float(params["u1y"]), float(params["u2y"])
    dxy = float(params["dxy"])
    nx, ny = int(params["nx"]), int(params["ny"])

    matrix = torch.eye(nx * ny, dtype=torch.float64) * -4.0
    for i in range(nx):
        for j in range(ny):
            index = i * ny + j
            if i > 0:
                matrix[index, (i - 1) * ny + j] = 1.0
            if i < nx - 1:
                matrix[index, (i + 1) * ny + j] = 1.0
            if j > 0:
                matrix[index, index - 1] = 1.0
            if j < ny - 1:
                matrix[index, index + 1] = 1.0

    forcing = torch.zeros((nx, ny), dtype=torch.float64)
    for i in range(nx):
        for j in range(ny):
            boundary = 0.0
            if i == 0:
                boundary += u1x * k / dxy**2
            if i == nx - 1:
                boundary += u2x * k / dxy**2
            if j == 0:
                boundary += u1y * k / dxy**2
            if j == ny - 1:
                boundary += u2y * k / dxy**2
            if boundary:
                forcing[i, j] = boundary
            elif 0 < i < nx - 1 and 0 < j < ny - 1:
                forcing[i, j] = 10.0 * np.sin(2.0 * np.pi * i * j / np.sqrt(nx * ny))

    rhs = -forcing.flatten() * dxy**2 / k
    matrix, rhs, scale_factor = _scale_system(matrix, rhs, scale)
    return {
        "name": "heat_2d",
        "matrix": matrix,
        "rhs": rhs,
        "matrix_role": "physical_hermitian_400x400",
        "scale_factor": scale_factor,
        "plot_shape": (nx, ny),
    }


def reconstruct_with_boundaries(
    interior: np.ndarray | torch.Tensor | list[float], x0: float, xT: float
) -> np.ndarray:
    """Add the two prescribed endpoints to an interior solution."""
    values = (
        interior.detach().cpu().numpy()
        if isinstance(interior, torch.Tensor)
        else np.asarray(interior)
    )
    return np.concatenate(([float(x0)], values, [float(xT)]))


def extract_damped_solution(
    embedded_solution: np.ndarray | torch.Tensor | list[float], n_interior: int
) -> np.ndarray:
    """Extract the physical solution from the embedding's second block."""
    solution = (
        embedded_solution.detach().cpu().numpy()
        if isinstance(embedded_solution, torch.Tensor)
        else np.asarray(embedded_solution)
    )
    return solution[n_interior:].copy()


def generate_random_problems(
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Generate the deterministic real symmetric random systems."""
    seed = int(config.get("seed", 12345))
    instance_count = int(config.get("instances", 20))
    dimension = int(config.get("dimension", 16))
    density = float(config.get("off_diagonal_density", 0.25))
    singular_tolerance = float(config.get("singular_tolerance", 1e-12))
    max_condition_number = float(config.get("max_condition_number", 10000.0))
    rng = np.random.default_rng(seed)
    upper_i, upper_j = np.triu_indices(dimension, k=1)
    nonzero_count = round(density * upper_i.size)
    problems: list[dict[str, object]] = []

    for instance_id in range(instance_count):
        for generation_attempt in range(1, 10001):
            rhs = rng.uniform(-1.0, 1.0, size=dimension)
            if np.linalg.norm(rhs) == 0.0:
                continue
            rhs /= np.linalg.norm(rhs)

            matrix = np.diag(rng.uniform(-1.0, 1.0, size=dimension))
            selected = rng.choice(upper_i.size, size=nonzero_count, replace=False)
            values = rng.uniform(-1.0, 1.0, size=nonzero_count)
            matrix[upper_i[selected], upper_j[selected]] = values
            matrix[upper_j[selected], upper_i[selected]] = values

            eigenvalues = np.linalg.eigvalsh(matrix)
            min_abs = float(np.min(np.abs(eigenvalues)))
            max_abs = float(np.max(np.abs(eigenvalues)))
            condition_number = max_abs / min_abs if min_abs > 0.0 else float("inf")
            if min_abs <= singular_tolerance:
                rejection_reason = "numerically_singular"
            elif condition_number > max_condition_number:
                rejection_reason = "condition_limit"
            else:
                rejection_reason = ""

            if rejection_reason:
                continue

            scale_factor = 1.01 * max_abs
            problems.append(
                {
                    "name": f"random_{instance_id:02d}",
                    "matrix": torch.as_tensor(
                        matrix / scale_factor, dtype=torch.float64
                    ),
                    "rhs": torch.as_tensor(rhs / scale_factor, dtype=torch.float64),
                    "matrix_role": "physical_hermitian_16x16",
                    "scale_factor": scale_factor,
                    "seed": seed,
                    "generation_attempt": generation_attempt,
                }
            )
            break
        else:
            raise RuntimeError(
                f"could not generate accepted random instance {instance_id}"
            )

    return problems
