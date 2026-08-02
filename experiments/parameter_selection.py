"""HHL spectral filter and reproducible ``(mu, tau)`` selection."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import torch


def _as_numpy(
    value: np.ndarray | torch.Tensor | list[float] | list[list[float]],
) -> np.ndarray:
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def signed_phase_indices(mu: int) -> np.ndarray:
    """Return signed representatives for phase bins ``0, ..., mu-1``."""
    bins = np.arange(mu, dtype=np.int64)
    return np.where(bins <= mu // 2, bins, bins - mu)


def dirichlet_kernel_squared(z: np.ndarray, mu: int) -> np.ndarray:
    """Evaluate the squared Dirichlet kernel with its removable limit."""
    values = np.asarray(z, dtype=np.float64)
    reduced = np.remainder(values + mu / 2.0, mu) - mu / 2.0
    return (mu * np.sinc(reduced) / np.sinc(reduced / mu)) ** 2


def hhl_filter(
    eigenvalues: np.ndarray,
    mu: int,
    tau: float,
    *,
    block_size: int = 256,
) -> np.ndarray:
    """Evaluate the Proposition 3.2 filter on Hermitian eigenvalues."""
    spectrum = np.asarray(eigenvalues, dtype=np.float64)

    signed_bins = signed_phase_indices(mu)
    weights = np.zeros(mu, dtype=np.float64)
    weights[signed_bins != 0] = 1.0 / signed_bins[signed_bins != 0]
    bins = np.arange(mu, dtype=np.float64)
    result = np.empty_like(spectrum)
    for start in range(0, spectrum.size, block_size):
        stop = min(start + block_size, spectrum.size)
        offsets = tau * spectrum[start:stop, None] - bins[None, :]
        result[start:stop] = (
            tau / mu**2 * (dirichlet_kernel_squared(offsets, mu) @ weights)
        )
    return result


def apply_spectral_filter(
    eigenvectors: np.ndarray,
    rhs: np.ndarray,
    filter_values: np.ndarray,
) -> np.ndarray:
    """Apply scalar filter values in a Hermitian eigenbasis."""
    coefficients = eigenvectors.conj().T @ rhs
    return eigenvectors @ (filter_values * coefficients)


def evaluate_parameter_candidate(
    *,
    problem: str,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    rhs: np.ndarray,
    mu: int,
    tau: float,
    zero_bin_separation: float,
    target_rhs_filter_relative_error: float,
) -> dict[str, object]:
    """Evaluate feasibility, filter errors, and TN cost for one candidate."""
    scaled = tau * eigenvalues
    min_abs_scaled = float(np.min(np.abs(scaled)))
    max_abs_scaled = float(np.max(np.abs(scaled)))
    no_aliasing = max_abs_scaled < mu / 2.0
    separated = min_abs_scaled >= zero_bin_separation

    filter_values = hhl_filter(eigenvalues, mu, tau)
    inverse_values = 1.0 / eigenvalues
    filtered_solution = apply_spectral_filter(eigenvectors, rhs, filter_values)
    reference_solution = apply_spectral_filter(eigenvectors, rhs, inverse_values)
    reference_norm = float(np.linalg.norm(reference_solution))

    rhs_error = float(
        np.linalg.norm(filtered_solution - reference_solution) / reference_norm
    )
    if not no_aliasing:
        rejection_reason = "aliasing"
    elif not separated:
        rejection_reason = "zero_bin_separation"
    else:
        rejection_reason = ""
    feasible = no_aliasing and separated
    dimension = eigenvalues.size

    return {
        "problem": problem,
        "mu": mu,
        "tau": float(tau),
        "feasible": feasible,
        "no_aliasing": no_aliasing,
        "min_abs_tau_lambda": min_abs_scaled,
        "max_abs_tau_lambda": max_abs_scaled,
        "max_grid_distance": float(np.max(np.abs(scaled - np.rint(scaled)))),
        "operator_absolute_filter_error": float(
            np.max(np.abs(filter_values - inverse_values))
        ),
        "operator_relative_filter_error": float(
            np.max(np.abs(eigenvalues * filter_values - 1.0))
        ),
        "rhs_relative_filter_error": rhs_error,
        "cost_proxy": int(dimension**2 * mu + dimension * mu**2 + mu**3),
        "target_met": bool(feasible and rhs_error <= target_rhs_filter_relative_error),
        "selected": False,
        "rejection_reason": rejection_reason,
    }


def choose_parameter_candidate(
    candidates: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Apply the target/cost rule and mark exactly one candidate."""
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        raise ValueError("parameter grid contains no feasible candidates")

    meeting_target = [candidate for candidate in feasible if candidate["target_met"]]
    if meeting_target:
        chosen = min(
            meeting_target,
            key=lambda item: (
                item["cost_proxy"],
                item["rhs_relative_filter_error"],
                item["mu"],
                item["tau"],
            ),
        )
    else:
        chosen = min(
            feasible,
            key=lambda item: (
                item["rhs_relative_filter_error"],
                item["cost_proxy"],
                item["mu"],
                item["tau"],
            ),
        )

    marked = [
        {**candidate, "selected": candidate is chosen} for candidate in candidates
    ]
    selected = next(candidate for candidate in marked if candidate["selected"])
    return selected, marked


def search_parameters(
    matrix: np.ndarray | torch.Tensor | list[list[float]],
    rhs: np.ndarray | torch.Tensor | list[float],
    problem: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the predefined logarithmic grids and select deterministically."""
    coefficient_matrix = _as_numpy(matrix)
    right_hand_side = _as_numpy(rhs)
    eigenvalues, eigenvectors = np.linalg.eigh(coefficient_matrix)
    absolute = np.abs(eigenvalues)
    min_abs = float(np.min(absolute))
    if min_abs <= float(config.get("singular_tolerance", 1e-12)):
        raise ValueError("matrix is numerically singular")

    max_abs = float(np.max(absolute))
    zero_separation = float(config["zero_bin_separation"])
    safety_factor = float(config["no_aliasing_safety_factor"])
    tau_points = int(config["tau_points_per_mu"])
    target = float(config["target_rhs_filter_relative_error"])
    candidates: list[dict[str, object]] = []

    for mu_value in config["mu_candidates"]:  # type: ignore[union-attr]
        mu = int(mu_value)
        tau_lower = zero_separation / min_abs
        tau_upper = safety_factor * mu / (2.0 * max_abs)
        if tau_lower >= tau_upper:
            continue
        for tau in np.geomspace(tau_lower, tau_upper, tau_points):
            candidates.append(
                evaluate_parameter_candidate(
                    problem=problem,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    rhs=right_hand_side,
                    mu=mu,
                    tau=float(tau),
                    zero_bin_separation=zero_separation,
                    target_rhs_filter_relative_error=target,
                )
            )

    selected, marked = choose_parameter_candidate(candidates)
    if not selected["target_met"]:
        warnings.warn(
            f"{problem}: filter-error target was not met; selected minimum-error feasible candidate",
            RuntimeWarning,
            stacklevel=2,
        )
    return {"selected": selected, "candidates": marked}
