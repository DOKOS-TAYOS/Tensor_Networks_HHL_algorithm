"""Generate all reproducible artifacts for Referee 1 comments 5 and 6."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from experiments.parameter_selection import (
    apply_spectral_filter,
    hhl_filter,
    search_parameters,
)
from experiments.problem_builders import (
    build_damped_oscillator,
    build_harmonic_oscillator,
    build_heat_problem,
    extract_damped_solution,
    generate_random_problems,
    reconstruct_with_boundaries,
)
from tn_hhl import tensornetwork_HHL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config_r1_c5_c6.json")
DEFAULT_OUTPUT = ROOT / "artifacts" / "reviewer_r1_c5_c6"


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Root mean squared error between two solution vectors."""
    difference = _to_numpy(estimate) - _to_numpy(reference)
    return float(np.sqrt(np.mean(np.abs(difference) ** 2)))


def relative_solution_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Euclidean solution error relative to the reference norm."""
    estimate_array = _to_numpy(estimate)
    reference_array = _to_numpy(reference)
    return float(
        np.linalg.norm(estimate_array - reference_array)
        / np.linalg.norm(reference_array)
    )


def normalized_residual(
    matrix: np.ndarray, estimate: np.ndarray, rhs: np.ndarray
) -> float:
    """Residual norm relative to the right-hand-side norm."""
    matrix_array = _to_numpy(matrix)
    estimate_array = _to_numpy(estimate)
    rhs_array = _to_numpy(rhs)
    return float(
        np.linalg.norm(matrix_array @ estimate_array - rhs_array)
        / np.linalg.norm(rhs_array)
    )


def spectral_diagnostics(
    matrix: np.ndarray | torch.Tensor,
    tau: float,
    mu: int,
    singular_tolerance: float = 1e-12,
    grid_tolerance: float = 1e-10,
) -> tuple[dict[str, object], np.ndarray]:
    """Compute the spectrum and its position on the HHL phase grid."""
    eigenvalues = np.linalg.eigvalsh(_to_numpy(matrix))
    absolute = np.abs(eigenvalues)
    min_abs = float(np.min(absolute))
    if min_abs <= singular_tolerance:
        raise ValueError("matrix is numerically singular")

    scaled = tau * eigenvalues
    grid_distances = np.abs(scaled - np.rint(scaled))
    max_abs_scaled = float(np.max(np.abs(scaled)))
    diagnostics = {
        "dimension": len(eigenvalues),
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
        "max_abs_lambda": float(np.max(absolute)),
        "min_abs_lambda": min_abs,
        "condition_number": float(np.max(absolute)) / min_abs,
        "tau_lambda_min": float(np.min(scaled)),
        "tau_lambda_max": float(np.max(scaled)),
        "min_abs_tau_lambda": float(np.min(np.abs(scaled))),
        "max_abs_tau_lambda": max_abs_scaled,
        "max_grid_distance": float(np.max(grid_distances)),
        "mean_grid_distance": float(np.mean(grid_distances)),
        "no_aliasing": max_abs_scaled < mu / 2.0,
        "aliasing_margin": mu / 2.0 - max_abs_scaled,
        "aliasing_ratio": 2.0 * max_abs_scaled / mu,
        "exact_phase_grid": bool(np.all(grid_distances < grid_tolerance)),
    }
    return diagnostics, eigenvalues


def tau_lambda_rows(
    problem: str,
    matrix_role: str,
    eigenvalues: np.ndarray,
    tau: float,
    mu: int,
) -> list[dict[str, object]]:
    """Return one row per eigenvalue for the phase-grid diagnostic."""
    rows = []
    for index, eigenvalue in enumerate(eigenvalues):
        scaled = tau * float(eigenvalue)
        nearest = round(scaled)
        rows.append(
            {
                "problem": problem,
                "matrix_role": matrix_role,
                "eigenvalue_index": index,
                "eigenvalue": float(eigenvalue),
                "tau": tau,
                "tau_lambda": scaled,
                "nearest_integer": nearest,
                "grid_distance": abs(scaled - nearest),
                "aliased": abs(scaled) >= mu / 2.0,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_statistics(
    values: Sequence[float],
) -> tuple[float, float, float, float, float]:
    """Return mean, sample deviation, standard error, and 95% Student-t interval."""
    sample = np.asarray(values, dtype=float)
    mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    standard_error = sample_std / np.sqrt(sample.size)
    half_width = float(stats.t.ppf(0.975, df=sample.size - 1)) * standard_error
    return mean, sample_std, standard_error, mean - half_width, mean + half_width


def log_scale_confidence_errors(
    means: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build asymmetric error bars, omitting lower arms that cannot be logged."""
    omitted = lows <= 0
    lower_errors = np.where(omitted, 0.0, means - lows)
    upper_errors = highs - means
    return np.vstack((lower_errors, upper_errors)), omitted


def _measure_time(
    function: Callable[[], object], repetitions: int
) -> tuple[object, float]:
    """Return the result and median runtime after one warm-up."""
    if repetitions < 1:
        raise ValueError("timing requires at least one repetition")
    if repetitions > 1:
        function()  # Warm-up repeated solver benchmarks only.
    durations: list[float] = []
    result: object | None = None
    for _ in range(repetitions):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    assert result is not None
    return result, float(np.median(durations))


def compare_tn_and_filter(
    *,
    problem: Mapping[str, object],
    instance_id: int,
    n_c: int,
    tau: float,
) -> dict[str, object]:
    """Compare a complete TN contraction with the equivalent spectral filter."""
    mu = 2**n_c
    matrix = _to_numpy(problem["matrix"])
    rhs = _to_numpy(problem["rhs"])

    def spectral_solution() -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        filter_values = hhl_filter(eigenvalues, mu, tau)
        return apply_spectral_filter(eigenvectors, rhs, filter_values).real

    filter_estimate, filter_time = _measure_time(spectral_solution, 1)
    tn_tensor, tn_time = _measure_time(
        lambda: tensornetwork_HHL(mu, tau, problem["rhs"], problem["matrix"]), 1
    )
    tn_estimate = _to_numpy(tn_tensor).real
    reference = np.linalg.solve(matrix, rhs)
    filter_norm = float(np.linalg.norm(filter_estimate))
    if filter_norm == 0.0:
        raise ValueError("spectral-filter solution norm must be non-zero")
    diagnostics, _ = spectral_diagnostics(
        problem["matrix"],
        tau,
        mu,
    )
    return {
        "problem": problem["name"],
        "instance_id": instance_id,
        "n_c": n_c,
        "mu": mu,
        "tau": tau,
        "condition_number": diagnostics["condition_number"],
        "no_aliasing": diagnostics["no_aliasing"],
        "evaluation_pair": "full_tn_vs_spectral_filter",
        "tn_filter_rmse": rmse(tn_estimate, filter_estimate),
        "tn_filter_relative_difference": float(
            np.linalg.norm(tn_estimate - filter_estimate) / filter_norm
        ),
        "tn_solution_relative_error": relative_solution_error(tn_estimate, reference),
        "filter_solution_relative_error": relative_solution_error(
            filter_estimate, reference
        ),
        "tn_normalized_residual": normalized_residual(matrix, tn_estimate, rhs),
        "filter_normalized_residual": normalized_residual(matrix, filter_estimate, rhs),
        "tn_time_seconds": tn_time,
        "spectral_filter_time_seconds": filter_time,
        "tn_preprocessing_U_included": True,
        "spectral_filter_eigendecomposition_included": True,
    }


def _save_figure(figure: plt.Figure, path_without_suffix: Path) -> None:
    figure.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "Tensor_Networks_HHL_algorithm",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _plot_oscillator(
    path: Path,
    t_nodes: np.ndarray,
    reference: np.ndarray,
    estimate: np.ndarray,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.plot(t_nodes, reference, color="tab:blue", linewidth=2.3, label="Direct solve")
    axis.plot(
        t_nodes,
        estimate,
        color="tab:red",
        linestyle="none",
        marker=".",
        markersize=5,
        label="TN HHL",
    )
    axis.set(xlabel="t", ylabel="x(t)", title=title)
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def _plot_heat(
    output_dir: Path,
    estimate: np.ndarray,
    reference: np.ndarray,
    params: Mapping[str, object],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    coordinate = np.arange(estimate.size, dtype=float) * float(params["dxy"])
    axis.plot(
        coordinate, reference, color="tab:blue", linewidth=1.5, label="Direct solve"
    )
    axis.plot(
        coordinate,
        estimate,
        color="tab:red",
        linestyle="none",
        marker=".",
        markersize=3,
        label="TN HHL",
    )
    axis.set(
        xlabel="Flattened grid coordinate",
        ylabel="Temperature",
        title="Two-dimensional static heat equation",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    _save_figure(figure, output_dir / "C2D")

    nx, ny = int(params["nx"]), int(params["ny"])
    full = np.empty((nx + 2, ny + 2), dtype=float)
    full[1:-1, 1:-1] = estimate.reshape(nx, ny)
    full[0, :], full[-1, :] = float(params["u1x"]), float(params["u2x"])
    full[:, 0], full[:, -1] = float(params["u1y"]), float(params["u2y"])
    full[0, 0] = 0.5 * (float(params["u1x"]) + float(params["u1y"]))
    full[0, -1] = 0.5 * (float(params["u1x"]) + float(params["u2y"]))
    full[-1, 0] = 0.5 * (float(params["u2x"]) + float(params["u1y"]))
    full[-1, -1] = 0.5 * (float(params["u2x"]) + float(params["u2y"]))
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    image = axis.pcolormesh(full, cmap="CMRmap", shading="auto")
    figure.colorbar(image, ax=axis, label="Temperature")
    axis.set(
        xlabel="y grid index",
        ylabel="x grid index",
        title="TN HHL heat solution with prescribed boundaries",
    )
    _save_figure(figure, output_dir / "C2D_2D")


def _run_application_problem(
    *,
    problem: Mapping[str, object],
    hhl_matrix: torch.Tensor,
    hhl_rhs: torch.Tensor,
    extract: Callable[[np.ndarray], np.ndarray],
    config: Mapping[str, object],
    timing_repetitions: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
]:
    problem_name = str(problem["name"])
    matrix_role = str(problem["matrix_role"])
    physical_matrix = problem["matrix"]
    physical_rhs = problem["rhs"]
    selection, selection_time = _measure_time(
        lambda: search_parameters(hhl_matrix, hhl_rhs, problem_name, config), 1
    )
    selected = selection["selected"]
    reference_tensor, reference_time = _measure_time(
        lambda: torch.linalg.solve(physical_matrix, physical_rhs), timing_repetitions
    )
    embedded_estimate, tn_time = _measure_time(
        lambda: tensornetwork_HHL(selected["mu"], selected["tau"], hhl_rhs, hhl_matrix),
        timing_repetitions,
    )
    reference = _to_numpy(reference_tensor).real
    estimate = extract(_to_numpy(embedded_estimate).real)
    diagnostics, eigenvalues = spectral_diagnostics(
        hhl_matrix,
        selected["tau"],
        selected["mu"],
        singular_tolerance=float(config["singular_tolerance"]),
        grid_tolerance=float(config["grid_tolerance"]),
    )
    result = {
        "problem": problem_name,
        "matrix_role": matrix_role,
        "physical_dimension": int(physical_matrix.shape[0]),
        "hhl_dimension": int(hhl_matrix.shape[0]),
        "mu": selected["mu"],
        "tau": selected["tau"],
        "target_met": selected["target_met"],
        "parameter_selection_method": str(config["parameter_selection_method"]),
        "selection_time_seconds": selection_time,
        "predicted_rhs_filter_relative_error": selected["rhs_relative_filter_error"],
        "rmse": rmse(estimate, reference),
        "relative_solution_error": relative_solution_error(estimate, reference),
        "normalized_residual": normalized_residual(
            physical_matrix, estimate, physical_rhs
        ),
        "tn_time_median_seconds": tn_time,
        "reference_time_median_seconds": reference_time,
        "tn_preprocessing_U_included": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    spectral_row = {
        "problem": problem_name,
        "matrix_role": matrix_role,
        "mu": selected["mu"],
        "tau": selected["tau"],
        **diagnostics,
    }
    phase_rows = tau_lambda_rows(
        problem=problem_name,
        matrix_role=matrix_role,
        eigenvalues=eigenvalues,
        tau=selected["tau"],
        mu=selected["mu"],
    )
    return result, selection, spectral_row, phase_rows, estimate, reference


def _application_experiments(
    config: Mapping[str, object], output_dir: Path
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    problem_configs = config["problems"]
    harmonic_params = problem_configs["harmonic_oscillator"]
    damped_params = problem_configs["damped_oscillator"]
    heat_params = problem_configs["heat_2d"]
    harmonic = build_harmonic_oscillator(harmonic_params, scale=True)
    damped = build_damped_oscillator(damped_params, scale=True)
    heat = build_heat_problem(heat_params, scale=False)
    repetitions = int(config["timing_repetitions"])
    selection_method = str(config["parameter_selection_method"])
    application_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    spectral_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []

    (
        harmonic_row,
        harmonic_selection,
        harmonic_spectrum,
        harmonic_phases,
        harmonic_estimate,
        harmonic_reference,
    ) = _run_application_problem(
        problem=harmonic,
        hhl_matrix=harmonic["matrix"],
        hhl_rhs=harmonic["rhs"],
        extract=lambda value: value,
        config=config,
        timing_repetitions=repetitions,
    )
    application_rows.append(harmonic_row)
    search_rows.extend(
        {**candidate, "parameter_selection_method": selection_method}
        for candidate in harmonic_selection["candidates"]
    )
    spectral_rows.append(harmonic_spectrum)
    phase_rows.extend(harmonic_phases)
    _plot_oscillator(
        output_dir / "OAF",
        harmonic["grid"]["t_nodes"],
        reconstruct_with_boundaries(harmonic_reference, harmonic["x0"], harmonic["xT"]),
        reconstruct_with_boundaries(harmonic_estimate, harmonic["x0"], harmonic["xT"]),
        "Forced harmonic oscillator (100 intervals, 99 interior unknowns)",
    )

    (
        damped_row,
        damped_selection,
        damped_spectrum,
        damped_phases,
        damped_estimate,
        damped_reference,
    ) = _run_application_problem(
        problem=damped,
        hhl_matrix=damped["embedding"],
        hhl_rhs=damped["embedded_rhs"],
        extract=lambda value: extract_damped_solution(
            value, damped["grid"]["n_interior"]
        ),
        config=config,
        timing_repetitions=repetitions,
    )
    application_rows.append(damped_row)
    search_rows.extend(
        {**candidate, "parameter_selection_method": selection_method}
        for candidate in damped_selection["candidates"]
    )
    spectral_rows.append(damped_spectrum)
    phase_rows.extend(damped_phases)
    _plot_oscillator(
        output_dir / "OAA",
        damped["grid"]["t_nodes"],
        reconstruct_with_boundaries(damped_reference, damped["x0"], damped["xT"]),
        reconstruct_with_boundaries(damped_estimate, damped["x0"], damped["xT"]),
        "Forced damped oscillator (198-dimensional Hermitian embedding)",
    )

    (
        heat_row,
        heat_selection,
        heat_spectrum,
        heat_phases,
        heat_estimate,
        heat_reference,
    ) = _run_application_problem(
        problem=heat,
        hhl_matrix=heat["matrix"],
        hhl_rhs=heat["rhs"],
        extract=lambda value: value,
        config=config,
        timing_repetitions=repetitions,
    )
    application_rows.append(heat_row)
    search_rows.extend(
        {**candidate, "parameter_selection_method": selection_method}
        for candidate in heat_selection["candidates"]
    )
    spectral_rows.append(heat_spectrum)
    phase_rows.extend(heat_phases)
    _plot_heat(output_dir, heat_estimate, heat_reference, heat_params)
    return application_rows, search_rows, spectral_rows, phase_rows


def _random_experiments(
    config: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    random_config = dict(config["random"])  # type: ignore[arg-type]
    random_config["singular_tolerance"] = config["singular_tolerance"]
    problems = generate_random_problems(random_config)
    result_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    spectral_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    repetitions = int(config["timing_repetitions"])
    selection_method = str(config["parameter_selection_method"])
    for instance_id, problem in enumerate(problems):
        selection, selection_time = _measure_time(
            lambda problem=problem: search_parameters(
                problem["matrix"], problem["rhs"], problem["name"], config
            ),
            1,
        )
        selected = selection["selected"]
        reference_tensor, reference_time = _measure_time(
            lambda problem=problem: torch.linalg.solve(
                problem["matrix"], problem["rhs"]
            ),
            repetitions,
        )
        estimate_tensor, tn_time = _measure_time(
            lambda problem=problem, selected=selected: tensornetwork_HHL(
                selected["mu"],
                selected["tau"],
                problem["rhs"],
                problem["matrix"],
            ),
            repetitions,
        )
        reference = _to_numpy(reference_tensor).real
        estimate = _to_numpy(estimate_tensor).real
        diagnostics, eigenvalues = spectral_diagnostics(
            problem["matrix"],
            selected["tau"],
            selected["mu"],
            singular_tolerance=float(config["singular_tolerance"]),
            grid_tolerance=float(config["grid_tolerance"]),
        )
        result_rows.append(
            {
                "instance_id": instance_id,
                "seed": int(problem["seed"]),
                "generation_attempt": int(problem["generation_attempt"]),
                "lambda_min": diagnostics["lambda_min"],
                "lambda_max": diagnostics["lambda_max"],
                "min_abs_lambda": diagnostics["min_abs_lambda"],
                "condition_number": diagnostics["condition_number"],
                "rmse": rmse(estimate, reference),
                "relative_solution_error": relative_solution_error(estimate, reference),
                "normalized_residual": normalized_residual(
                    problem["matrix"], estimate, problem["rhs"]
                ),
                "mu": selected["mu"],
                "tau": selected["tau"],
                "no_aliasing": diagnostics["no_aliasing"],
                "max_grid_distance": diagnostics["max_grid_distance"],
                "predicted_rhs_filter_relative_error": selected[
                    "rhs_relative_filter_error"
                ],
                "parameter_selection_method": selection_method,
                "selection_time_seconds": selection_time,
                "tn_time_median_seconds": tn_time,
                "reference_time_median_seconds": reference_time,
                "tn_preprocessing_U_included": True,
            }
        )
        search_rows.extend(
            {**candidate, "parameter_selection_method": selection_method}
            for candidate in selection["candidates"]
        )
        spectral_rows.append(
            {
                "problem": problem["name"],
                "matrix_role": problem["matrix_role"],
                "mu": selected["mu"],
                "tau": selected["tau"],
                **diagnostics,
            }
        )
        phase_rows.extend(
            tau_lambda_rows(
                problem=problem["name"],
                matrix_role=problem["matrix_role"],
                eigenvalues=eigenvalues,
                tau=selected["tau"],
                mu=selected["mu"],
            )
        )
    return problems, result_rows, search_rows, spectral_rows, phase_rows


def _hyperparameter_sweep(
    problems: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    output_dir: Path,
) -> list[dict[str, object]]:
    sweep_config = config["hyperparameter_sweep"]
    tau_values = [float(value) for value in sweep_config["tau_values"]]  # type: ignore[index]
    n_c_values = [int(value) for value in sweep_config["n_c_values"]]  # type: ignore[index]
    caches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for problem in problems:
        matrix = _to_numpy(problem["matrix"])
        rhs = _to_numpy(problem["rhs"])
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        reference = np.linalg.solve(matrix, rhs)
        caches.append((matrix, rhs, eigenvalues, eigenvectors, reference))
    rows: list[dict[str, object]] = []
    for n_c in n_c_values:
        mu = 2**n_c
        for tau in tau_values:
            rmses: list[float] = []
            relative_errors: list[float] = []
            residuals: list[float] = []
            no_aliasing_flags: list[float] = []
            mean_grid_distances: list[float] = []
            max_grid_distances: list[float] = []
            for matrix, rhs, eigenvalues, eigenvectors, reference in caches:
                filter_values = hhl_filter(eigenvalues, mu, tau)
                estimate = apply_spectral_filter(eigenvectors, rhs, filter_values).real
                scaled = tau * eigenvalues
                distances = np.abs(scaled - np.rint(scaled))
                rmses.append(rmse(estimate, reference))
                relative_errors.append(relative_solution_error(estimate, reference))
                residuals.append(normalized_residual(matrix, estimate, rhs))
                no_aliasing_flags.append(float(np.max(np.abs(scaled)) < mu / 2.0))
                mean_grid_distances.append(float(np.mean(distances)))
                max_grid_distances.append(float(np.max(distances)))
            rmse_mean, rmse_std, rmse_se, rmse_low, rmse_high = mean_statistics(rmses)
            rows.append(
                {
                    "n_c": n_c,
                    "mu": mu,
                    "tau": tau,
                    "instance_count": len(problems),
                    "rmse_mean": rmse_mean,
                    "rmse_sample_std": rmse_std,
                    "rmse_standard_error": rmse_se,
                    "rmse_confidence_interval_95_low": rmse_low,
                    "rmse_confidence_interval_95_high": rmse_high,
                    "relative_solution_error_mean": float(np.mean(relative_errors)),
                    "normalized_residual_mean": float(np.mean(residuals)),
                    "no_aliasing_fraction": float(np.mean(no_aliasing_flags)),
                    "mean_grid_distance": float(np.mean(mean_grid_distances)),
                    "max_grid_distance": float(np.max(max_grid_distances)),
                    "confidence_interval_method": "Student t, two-sided 95% for the mean",
                    "evaluation_method": str(sweep_config["evaluation_method"]),
                }
            )
    _plot_sweep(rows, n_c_values, tau_values, output_dir)
    return rows


def _tn_filter_validation(
    problems: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> list[dict[str, object]]:
    """Run full TN on the configured subset used to validate the spectral sweep."""
    validation = config["tn_filter_validation"]
    instance_ids = [int(value) for value in validation["instance_ids"]]  # type: ignore[index]
    points = validation["points"]
    rows: list[dict[str, object]] = []
    for instance_id in instance_ids:
        problem = problems[instance_id]
        for point in points:
            rows.append(
                compare_tn_and_filter(
                    problem=problem,
                    instance_id=instance_id,
                    n_c=int(point["n_c"]),
                    tau=float(point["tau"]),
                )
            )
    return rows


def _plot_sweep(
    rows: Sequence[Mapping[str, object]],
    n_c_values: Sequence[int],
    tau_values: Sequence[float],
    output_dir: Path,
) -> None:
    lookup = {(int(row["n_c"]), float(row["tau"])): row for row in rows}
    all_means = np.array([float(row["rmse_mean"]) for row in rows])
    plot_floor = 0.05 * float(np.min(all_means))
    figure, axis = plt.subplots(figsize=(9, 6.5))
    omitted_lower_count = 0
    for n_c in n_c_values:
        means = np.array([float(lookup[(n_c, tau)]["rmse_mean"]) for tau in tau_values])
        lows = np.array(
            [
                float(lookup[(n_c, tau)]["rmse_confidence_interval_95_low"])
                for tau in tau_values
            ]
        )
        highs = np.array(
            [
                float(lookup[(n_c, tau)]["rmse_confidence_interval_95_high"])
                for tau in tau_values
            ]
        )
        errors, omitted = log_scale_confidence_errors(means, lows, highs)
        omitted_lower_count += int(np.count_nonzero(omitted))
        error_bars = axis.errorbar(
            tau_values,
            means,
            yerr=errors,
            marker="o",
            markersize=3,
            capsize=2,
            label=rf"$n_c={n_c}$",
        )
        aliased = np.array(
            [
                float(lookup[(n_c, tau)]["no_aliasing_fraction"]) < 1.0
                for tau in tau_values
            ]
        )
        axis.scatter(
            np.asarray(tau_values)[aliased],
            means[aliased],
            color=error_bars.lines[0].get_color(),
            marker="x",
            s=28,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(bottom=plot_floor)
    axis.set(
        xlabel=r"$\tau$",
        ylabel="Mean RMSE",
        title="Error bars: two-sided 95% Student-t confidence interval for the mean",
    )
    axis.text(
        0.01,
        0.01,
        f"Lower arms omitted for {omitted_lower_count}/{len(rows)} intervals whose "
        "lower bound is nonpositive; CSV values are exact. Crosses mark aliasing.",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize=8, ncol=2)
    _save_figure(figure, output_dir / "rmse_vs_tau")

    figure, axis = plt.subplots(figsize=(9, 6.5))
    for tau in tau_values:
        means = np.array([float(lookup[(n_c, tau)]["rmse_mean"]) for n_c in n_c_values])
        lows = np.array(
            [
                float(lookup[(n_c, tau)]["rmse_confidence_interval_95_low"])
                for n_c in n_c_values
            ]
        )
        highs = np.array(
            [
                float(lookup[(n_c, tau)]["rmse_confidence_interval_95_high"])
                for n_c in n_c_values
            ]
        )
        clipped_lows = np.maximum(lows, plot_floor)
        errors = np.vstack((means - clipped_lows, highs - means))
        axis.errorbar(
            n_c_values,
            means,
            yerr=errors,
            marker="o",
            markersize=3,
            capsize=2,
            label=rf"$\tau={tau:g}$",
        )
    axis.set_yscale("log")
    axis.set_ylim(bottom=plot_floor)
    axis.set(
        xlabel=r"$n_c$ ($\mu=2^{n_c}$)",
        ylabel="Mean RMSE",
        title="Error bars: two-sided 95% Student-t confidence interval for the mean",
    )
    axis.text(
        0.01,
        0.01,
        "Nonpositive lower bounds are clipped to the plot floor; CSV values are exact.",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize=7, ncol=2)
    _save_figure(figure, output_dir / "rmse_vs_num_anc")


def run_experiments(
    config_path: Path = DEFAULT_CONFIG, output_dir: Path = DEFAULT_OUTPUT
) -> None:
    """Run the reviewer experiments and write their scientific results."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    application_rows, search_rows, spectral_rows, phase_rows = _application_experiments(
        config, output_dir
    )
    (
        random_problems,
        random_rows,
        random_search_rows,
        random_spectral_rows,
        random_phase_rows,
    ) = _random_experiments(config)
    search_rows.extend(random_search_rows)
    spectral_rows.extend(random_spectral_rows)
    phase_rows.extend(random_phase_rows)
    sweep_rows = _hyperparameter_sweep(random_problems, config, output_dir)
    tn_filter_validation_rows = _tn_filter_validation(random_problems, config)

    _write_csv(output_dir / "application_results.csv", application_rows)
    _write_csv(output_dir / "spectral_diagnostics.csv", spectral_rows)
    _write_csv(output_dir / "tau_lambda_values.csv", phase_rows)
    _write_csv(output_dir / "parameter_search.csv", search_rows)
    _write_csv(output_dir / "random_instance_results.csv", random_rows)
    _write_csv(output_dir / "hyperparameter_sweep.csv", sweep_rows)
    _write_csv(output_dir / "tn_filter_validation.csv", tn_filter_validation_rows)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_experiments(args.config, args.output_dir)
    print(f"Generated scientific results in {args.output_dir}")


if __name__ == "__main__":
    main()
