"""Regenerate the exact Qiskit--TN comparison for Referee 1 comment 7."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.parameter_selection import search_parameters
from experiments.problem_builders import generate_random_problems
from quantum_hhl import (
    build_hhl_unitary,
    controlled_rotation_angles,
    run_qiskit_hhl_once,
)
from tn_hhl import tensornetwork_HHL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config_r1_c5_c6.json")

QISKIT_TIMING_KEYS = (
    "unitary_seconds",
    "circuit_seconds",
    "transpile_seconds",
    "statevector_seconds",
    "extraction_seconds",
    "shots_seconds",
    "total_exact_seconds",
)
TN_TIMING_KEYS = (
    "unitary_seconds",
    "preparation_seconds",
    "contraction_seconds",
    "postprocessing_seconds",
    "total_seconds",
)


def normalized_state_and_probabilities(
    vector: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a vector and return its computational probabilities."""
    values = (
        vector.detach().cpu().numpy()
        if isinstance(vector, torch.Tensor)
        else np.asarray(vector)
    )
    state = np.asarray(values, dtype=np.complex128)
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise ValueError("state norm must be nonzero")
    state = state / norm
    return state, np.abs(state) ** 2


def probability_rmse(first: np.ndarray, second: np.ndarray) -> float:
    """Return component-wise RMSE between probability vectors."""
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def pure_state_fidelity(first: np.ndarray, second: np.ndarray) -> float:
    """Return squared overlap between normalized pure states."""
    value = float(np.abs(np.vdot(first, second)) ** 2)
    if not -1e-12 <= value <= 1.0 + 1e-12:
        raise ValueError(f"fidelity outside [0, 1]: {value}")
    return min(max(value, 0.0), 1.0)


def pure_mixed_fidelity(state: np.ndarray, density_matrix: np.ndarray) -> float:
    """Return ``<state|density_matrix|state>``."""
    value = float(np.vdot(state, density_matrix @ state).real)
    if not -1e-12 <= value <= 1.0 + 1e-12:
        raise ValueError(f"fidelity outside [0, 1]: {value}")
    return min(max(value, 0.0), 1.0)


def density_matrix_purity(density_matrix: np.ndarray) -> float:
    """Return ``Tr(rho**2)`` for the ancilla-conditioned state."""
    value = float(np.trace(density_matrix @ density_matrix).real)
    if not -1e-12 <= value <= 1.0 + 1e-12:
        raise ValueError(f"purity outside [0, 1]: {value}")
    return min(max(value, 0.0), 1.0)


def run_tn_hhl_once(
    mu: int,
    tau: float,
    b_vector: torch.Tensor,
    A_matrix: torch.Tensor,
) -> dict[str, object]:
    """Run one TN contraction with the existing timing phases."""
    phase_timings: dict[str, float] = {}
    result = tensornetwork_HHL(
        mu,
        tau,
        b_vector,
        A_matrix,
        timings=phase_timings,
    )
    start = perf_counter()
    state, probabilities = normalized_state_and_probabilities(result)
    postprocessing_seconds = perf_counter() - start
    return {
        "result": result,
        "state": state,
        "probabilities": probabilities,
        **phase_timings,
        "postprocessing_seconds": postprocessing_seconds,
        "total_seconds": sum(phase_timings.values()) + postprocessing_seconds,
    }


def median_timings(
    runs: list[dict[str, object]],
    keys: tuple[str, ...],
) -> dict[str, float]:
    """Take the per-phase median across measured repetitions."""
    return {key: float(np.median([float(run[key]) for run in runs])) for key in keys}


def comparison_selection_config(config: Mapping[str, object]) -> dict[str, object]:
    """Combine the common search grid with the circuit-specific settings."""
    comparison = config["qiskit_comparison"]
    if not isinstance(comparison, Mapping):
        raise TypeError("qiskit_comparison must be a mapping")
    return {
        **config,
        "mu_candidates": comparison["mu_candidates"],
        "controlled_rotation_safety_factor": comparison[
            "controlled_rotation_safety_factor"
        ],
    }


def select_reviewer_instances(
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Regenerate and preflight all configured reviewer instances."""
    random_config = dict(config["random"])  # type: ignore[arg-type]
    random_config["singular_tolerance"] = config["singular_tolerance"]
    problems = generate_random_problems(random_config)
    selection_config = comparison_selection_config(config)
    selected_instances: list[dict[str, object]] = []
    failures: list[str] = []

    for instance, problem in enumerate(problems):
        start = perf_counter()
        selected = search_parameters(
            problem["matrix"],
            problem["rhs"],
            str(problem["name"]),
            selection_config,
        )["selected"]
        selection_seconds = perf_counter() - start
        mu = int(selected["mu"])
        tau = float(selected["tau"])
        c_phys = float(selected["C_phys"])
        matrix = problem["matrix"]
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("reviewer matrices must be torch tensors")
        scaled = tau * np.linalg.eigvalsh(matrix.detach().cpu().numpy())
        checks = {
            "no_aliasing": bool(np.max(np.abs(scaled)) < mu / 2.0),
            "zero_bin_separated": bool(
                np.min(np.abs(scaled)) > float(config["zero_bin_separation"])
            ),
            "rotation_valid": bool(float(selected["C_bin"]) <= 1.0),
            "parameter_target_met": bool(selected["target_met"]),
        }
        try:
            controlled_rotation_angles(mu, tau, c_phys)
        except ValueError as error:
            checks["rotation_valid"] = False
            failures.append(f"instance {instance}: {error}")
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            failures.append(
                f"instance {instance}: failed {', '.join(failed_checks)}; "
                f"mu={mu}, tau={tau:.16g}, "
                f"rhs_relative_filter_error="
                f"{float(selected['rhs_relative_filter_error']):.16g}"
            )
        selected_instances.append(
            {
                "instance": instance,
                "problem": problem,
                "selected": selected,
                "selection_seconds": selection_seconds,
            }
        )

    if failures:
        raise ValueError(
            "reviewer comparison preflight failed:\n" + "\n".join(failures)
        )
    return selected_instances


def benchmark_problem(
    instance: int,
    problem: Mapping[str, object],
    selected: Mapping[str, object],
    selection_seconds: float,
    comparison: Mapping[str, object],
) -> dict[str, object]:
    """Benchmark one common finite HHL discretization."""
    matrix = problem["matrix"]
    rhs = problem["rhs"]
    if not isinstance(matrix, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("benchmark problems must contain torch tensors")
    mu = int(selected["mu"])
    tau = float(selected["tau"])
    c_phys = float(selected["C_phys"])
    c_bin = float(selected["C_bin"])
    n_ancillas = mu.bit_length() - 1
    scaled = tau * np.linalg.eigvalsh(matrix.detach().cpu().numpy())
    no_aliasing = bool(np.max(np.abs(scaled)) < mu / 2.0)
    singular_bin_assigned = bool(np.any(np.rint(scaled) == 0.0))
    zero_bin_separated = bool(float(selected["min_abs_tau_lambda"]) > 0.5)
    rotation_valid = bool(c_bin <= 1.0)

    qiskit_unitary = build_hhl_unitary(matrix, tau, mu)
    phase_scale = 2.0 * np.pi * tau / mu
    tn_unitary = (
        torch.matrix_exp(1j * phase_scale * matrix.to(dtype=torch.complex128))
        .detach()
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(qiskit_unitary, tn_unitary, atol=1e-11, rtol=1e-11)
    controlled_rotation_angles(mu, tau, c_phys)

    qiskit_arguments = {
        "n_ancillas": n_ancillas,
        "b_vector": rhs,
        "A_matrix": matrix,
        "tau": tau,
        "c_phys": c_phys,
        "n_shots": int(comparison["shots"]),
        "seed_transpiler": int(comparison["seed_transpiler"]),
        "seed_simulator": int(comparison["seed_simulator"]),
        "threads": int(comparison["threads"]),
    }
    run_qiskit_hhl_once(**qiskit_arguments)
    run_tn_hhl_once(mu, tau, rhs, matrix)

    repetitions = int(comparison["timing_repetitions"])
    qiskit_runs: list[dict[str, object]] = []
    tn_runs: list[dict[str, object]] = []
    for _ in range(repetitions):
        qiskit_runs.append(run_qiskit_hhl_once(**qiskit_arguments))
        tn_runs.append(run_tn_hhl_once(mu, tau, rhs, matrix))

    qiskit_timing = median_timings(qiskit_runs, QISKIT_TIMING_KEYS)
    tn_timing = median_timings(tn_runs, TN_TIMING_KEYS)
    qiskit_result = qiskit_runs[-1]
    tn_result = tn_runs[-1]
    density_matrix = np.asarray(qiskit_result["ancilla_conditioned_density_matrix"])
    clock_zero_state = np.asarray(qiskit_result["clock_zero_solution_state"])
    ancilla_probabilities = np.asarray(
        qiskit_result["ancilla_conditioned_probabilities"]
    )
    clock_zero_probabilities = np.asarray(
        qiskit_result["ancilla_clock_zero_probabilities"]
    )
    sampled_probabilities = np.asarray(
        qiskit_result["sampled_ancilla_conditioned_probabilities"]
    )
    tn_state = np.asarray(tn_result["state"])
    tn_probabilities = np.asarray(tn_result["probabilities"])
    tn_vector = np.asarray(tn_result["result"])
    rhs_norm_squared = float(torch.linalg.vector_norm(rhs).item() ** 2)
    tn_joint_probability = float(
        c_phys**2 * np.linalg.norm(tn_vector) ** 2 / rhs_norm_squared
    )
    qiskit_joint_probability = float(qiskit_result["ancilla_clock_zero_probability"])
    joint_probability_difference = abs(qiskit_joint_probability - tn_joint_probability)
    np.testing.assert_allclose(
        qiskit_joint_probability,
        tn_joint_probability,
        atol=1e-10,
        rtol=1e-8,
    )

    return {
        "instance": instance,
        "seed": int(problem["seed"]),
        "generation_attempt": int(problem["generation_attempt"]),
        "dimension": int(matrix.shape[0]),
        "dtype": "complex128",
        "threads": int(comparison["threads"]),
        "mu": mu,
        "tau": tau,
        "C_phys": c_phys,
        "C_bin": c_bin,
        "no_aliasing": no_aliasing,
        "aliasing_margin": float(mu / 2.0 - np.max(np.abs(scaled))),
        "singular_bin_assigned": singular_bin_assigned,
        "zero_bin_separated": zero_bin_separated,
        "rotation_valid": rotation_valid,
        "parameter_target_met": bool(selected["target_met"]),
        "predicted_rhs_filter_relative_error": float(
            selected["rhs_relative_filter_error"]
        ),
        "parameter_selection_seconds": selection_seconds,
        "ancilla_success_probability": float(
            qiskit_result["ancilla_success_probability"]
        ),
        "ancilla_clock_zero_probability": qiskit_joint_probability,
        "tn_clock_zero_equivalent_probability": tn_joint_probability,
        "ancilla_clock_zero_probability_absolute_difference": (
            joint_probability_difference
        ),
        "fidelity_tn_ancilla_clock_zero": pure_state_fidelity(
            tn_state, clock_zero_state
        ),
        "fidelity_tn_ancilla": pure_mixed_fidelity(tn_state, density_matrix),
        "ancilla_conditioned_purity": density_matrix_purity(density_matrix),
        "probability_rmse_tn_ancilla_clock_zero": probability_rmse(
            tn_probabilities, clock_zero_probabilities
        ),
        "probability_rmse_tn_ancilla": probability_rmse(
            tn_probabilities, ancilla_probabilities
        ),
        "probability_rmse_sampled_ancilla_exact": probability_rmse(
            sampled_probabilities, ancilla_probabilities
        ),
        "sampled_ancilla_success_probability": float(
            qiskit_result["sampled_ancilla_success_probability"]
        ),
        "shots": int(comparison["shots"]),
        "successful_ancilla_shots": int(qiskit_result["successful_ancilla_shots"]),
        "timing_repetitions": repetitions,
        "qiskit_unitary_seconds": qiskit_timing["unitary_seconds"],
        "qiskit_circuit_seconds": qiskit_timing["circuit_seconds"],
        "qiskit_transpile_seconds": qiskit_timing["transpile_seconds"],
        "qiskit_statevector_seconds": qiskit_timing["statevector_seconds"],
        "qiskit_extraction_postprocessing_seconds": qiskit_timing["extraction_seconds"],
        "qiskit_shots_seconds": qiskit_timing["shots_seconds"],
        "qiskit_total_exact_seconds": qiskit_timing["total_exact_seconds"],
        "tn_unitary_seconds": tn_timing["unitary_seconds"],
        "tn_preparation_seconds": tn_timing["preparation_seconds"],
        "tn_contraction_seconds": tn_timing["contraction_seconds"],
        "tn_normalization_postprocessing_seconds": tn_timing["postprocessing_seconds"],
        "tn_total_seconds": tn_timing["total_seconds"],
        "speedup_total": (
            qiskit_timing["total_exact_seconds"] / tn_timing["total_seconds"]
        ),
        "speedup_core_valid": False,
        "speedup_core": "",
    }


def run_comparison(
    config_path: Path = DEFAULT_CONFIG,
    output_csv: Path | None = None,
) -> list[dict[str, object]]:
    """Regenerate all instances and write the versioned comparison CSV."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    comparison = config["qiskit_comparison"]
    if not isinstance(comparison, Mapping):
        raise TypeError("qiskit_comparison must be a mapping")
    torch.set_num_threads(int(comparison["threads"]))
    selected_instances = select_reviewer_instances(config)
    target_count = sum(
        bool(item["selected"]["target_met"])  # type: ignore[index]
        for item in selected_instances
    )
    print(
        f"{target_count}/{len(selected_instances)} instances with "
        "rhs_relative_filter_error <= 0.01"
    )

    rows: list[dict[str, object]] = []
    for item in selected_instances:
        row = benchmark_problem(
            int(item["instance"]),
            item["problem"],  # type: ignore[arg-type]
            item["selected"],  # type: ignore[arg-type]
            float(item["selection_seconds"]),
            comparison,
        )
        rows.append(row)
        print(
            f"Instance {int(row['instance']):02d}: mu={int(row['mu'])}, "
            f"tau={float(row['tau']):.6g}, "
            f"F_joint={float(row['fidelity_tn_ancilla_clock_zero']):.12f}"
        )

    destination = (
        output_csv if output_csv is not None else ROOT / str(comparison["output_csv"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} valid instances to {destination}")
    return rows


def main() -> None:
    """Parse command-line arguments and regenerate the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-csv", type=Path)
    arguments = parser.parse_args()
    run_comparison(arguments.config, arguments.output_csv)


if __name__ == "__main__":
    main()
