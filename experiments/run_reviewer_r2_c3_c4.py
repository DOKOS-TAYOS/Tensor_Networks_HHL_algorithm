"""Generate the memory and empirical-scaling results for Referee 2 C3/C4."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config_r1_c5_c6.json")
DEFAULT_COMPARISON_CSV = ROOT / "artifacts" / "reviewer_r1_c7_qiskit_comparison.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reviewer_r2_c3_c4"
MEMORY_COLUMNS = (
    "qiskit_exact_rss_baseline_bytes",
    "qiskit_exact_peak_rss_bytes",
    "qiskit_exact_peak_rss_delta_bytes",
    "tn_rss_baseline_bytes",
    "tn_peak_rss_bytes",
    "tn_peak_rss_delta_bytes",
)


def _load_config(config_path: Path) -> dict[str, object]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _random_config(config: Mapping[str, object]) -> dict[str, object]:
    values = dict(config["random"])  # type: ignore[arg-type]
    values["singular_tolerance"] = config["singular_tolerance"]
    return values


def _prepare_comparison_operation(
    operation: str,
    payload: Mapping[str, object],
) -> Callable[[], dict[str, float]]:
    import torch

    from experiments.problem_builders import generate_random_problems

    config = _load_config(Path(str(payload["config_path"])))
    problems = generate_random_problems(_random_config(config))
    problem = problems[int(payload["instance"])]
    mu = int(payload["mu"])
    tau = float(payload["tau"])

    if operation == "qiskit_exact":
        from quantum_hhl import run_qiskit_hhl_once

        comparison = config["qiskit_comparison"]  # type: ignore[assignment]

        def execute_qiskit() -> dict[str, float]:
            result = run_qiskit_hhl_once(
                n_ancillas=int(np.log2(mu)),
                b_vector=problem["rhs"],  # type: ignore[arg-type]
                A_matrix=problem["matrix"],  # type: ignore[arg-type]
                tau=tau,
                c_phys=float(payload["c_phys"]),
                n_shots=int(comparison["shots"]),  # type: ignore[index]
                seed_transpiler=int(
                    comparison["seed_transpiler"]  # type: ignore[index]
                ),
                seed_simulator=int(
                    comparison["seed_simulator"]  # type: ignore[index]
                ),
                threads=int(comparison["threads"]),  # type: ignore[index]
                include_shots=False,
            )
            density_matrix = np.asarray(result["density_matrix"])
            return {
                "result_norm": float(np.linalg.norm(density_matrix)),
                "success_probability": float(result["success_probability_exact"]),
            }

        return execute_qiskit

    if operation == "tn":
        from tn_hhl import tensornetwork_HHL

        def execute_tn() -> dict[str, float]:
            result = tensornetwork_HHL(
                mu,
                tau,
                problem["rhs"],  # type: ignore[arg-type]
                problem["matrix"],  # type: ignore[arg-type]
            )
            return {"result_norm": float(torch.linalg.vector_norm(result).item())}

        return execute_tn

    raise ValueError(f"unknown comparison operation: {operation}")


def _build_scaling_problem(
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dimension < 2:
        raise ValueError("scaling dimension must be at least two")
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(random_matrix)
    diagonal_signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    orthogonal *= diagonal_signs

    negative_count = dimension // 2
    positive_count = dimension - negative_count
    eigenvalues = np.concatenate(
        (
            np.linspace(-0.35, -0.10, negative_count),
            np.linspace(0.10, 0.35, positive_count),
        )
    )
    rng.shuffle(eigenvalues)
    matrix = orthogonal @ np.diag(eigenvalues) @ orthogonal.T
    matrix = 0.5 * (matrix + matrix.T)
    rhs = rng.normal(size=dimension)
    rhs /= np.linalg.norm(rhs)
    return matrix, rhs, eigenvalues


def _median_iqr(values: Sequence[float]) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(sample)),
        float(np.percentile(sample, 25.0)),
        float(np.percentile(sample, 75.0)),
    )


def _scaling_operation(payload: Mapping[str, object]) -> dict[str, object]:
    import torch

    from experiments.parameter_selection import (
        apply_spectral_filter,
        hhl_filter,
    )
    from tn_hhl import tensornetwork_HHL

    dimension = int(payload["N"])
    mu = int(payload["mu"])
    tau = float(payload["tau"])
    repetitions = int(payload["repetitions"])
    if repetitions < 1:
        raise ValueError("scaling repetitions must be positive")

    matrix, rhs, constructed_eigenvalues = _build_scaling_problem(
        dimension,
        int(payload["seed"]),
    )
    if not float(np.max(np.abs(tau * constructed_eigenvalues))) < mu / 2.0:
        raise ValueError(
            f"aliasing in scaling point N={dimension}, mu={mu}, tau={tau}"
        )
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float64)
    rhs_tensor = torch.as_tensor(rhs, dtype=torch.float64)

    start = time.perf_counter()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigendecomposition_seconds = time.perf_counter() - start

    start = time.perf_counter()
    filter_values = hhl_filter(eigenvalues, mu, tau)
    filter_values_seconds = time.perf_counter() - start

    start = time.perf_counter()
    filter_solution = apply_spectral_filter(eigenvectors, rhs, filter_values).real
    filter_application_seconds = time.perf_counter() - start

    warmup = tensornetwork_HHL(mu, tau, rhs_tensor, matrix_tensor)
    del warmup
    gc.collect()

    baseline_rss = psutil.Process().memory_info().rss
    connection = payload["connection"]
    start_event = payload["start_event"]
    connection.send({"stage": "baseline", "rss": baseline_rss})
    start_event.wait()

    total_times: list[float] = []
    unitary_times: list[float] = []
    preparation_times: list[float] = []
    contraction_times: list[float] = []
    estimate = None
    for _ in range(repetitions):
        timings: dict[str, float] = {}
        start = time.perf_counter()
        estimate = tensornetwork_HHL(
            mu,
            tau,
            rhs_tensor,
            matrix_tensor,
            timings=timings,
        )
        total_times.append(time.perf_counter() - start)
        unitary_times.append(timings["unitary_seconds"])
        preparation_times.append(timings["preparation_seconds"])
        contraction_times.append(timings["contraction_seconds"])

    assert estimate is not None
    estimate_array = estimate.detach().cpu().numpy()
    filter_norm = float(np.linalg.norm(filter_solution))
    if filter_norm == 0.0:
        raise ValueError("scaling filter solution has zero norm")
    total_median, total_q1, total_q3 = _median_iqr(total_times)
    unitary_median, _, _ = _median_iqr(unitary_times)
    preparation_median, _, _ = _median_iqr(preparation_times)
    contraction_median, _, _ = _median_iqr(contraction_times)
    return {
        "sweep": str(payload["sweep"]),
        "N": dimension,
        "mu": mu,
        "tau": tau,
        "repetitions": repetitions,
        "no_aliasing": True,
        "tn_total_median_seconds": total_median,
        "tn_total_q1_seconds": total_q1,
        "tn_total_q3_seconds": total_q3,
        "tn_unitary_median_seconds": unitary_median,
        "tn_preparation_median_seconds": preparation_median,
        "tn_contraction_median_seconds": contraction_median,
        "filter_eigendecomposition_seconds": eigendecomposition_seconds,
        "filter_values_seconds": filter_values_seconds,
        "filter_application_seconds": filter_application_seconds,
        "filter_total_seconds": (
            eigendecomposition_seconds
            + filter_values_seconds
            + filter_application_seconds
        ),
        "tn_filter_relative_difference": float(
            np.linalg.norm(estimate_array - filter_solution) / filter_norm
        ),
    }


def _isolated_worker(
    connection: Connection,
    start_event: Any,
    operation: str,
    payload: Mapping[str, object],
) -> None:
    try:
        if operation == "scaling":
            scaling_payload = {
                **payload,
                "connection": connection,
                "start_event": start_event,
            }
            result = _scaling_operation(scaling_payload)
        else:
            execute = _prepare_comparison_operation(operation, payload)
            warmup = execute()
            del warmup
            gc.collect()
            baseline_rss = psutil.Process().memory_info().rss
            connection.send({"stage": "baseline", "rss": baseline_rss})
            start_event.wait()
            result = execute()
        connection.send({"stage": "result", "result": result})
    except BaseException:
        connection.send({"stage": "error", "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _receive_message(
    connection: Connection,
    timeout_seconds: float = 0.0,
) -> dict[str, object] | None:
    try:
        if not connection.poll(timeout_seconds):
            return None
        return dict(connection.recv())
    except (BrokenPipeError, EOFError, OSError):
        return None


def _run_isolated(
    operation: str,
    payload: Mapping[str, object],
    *,
    baseline_timeout_seconds: float = 600.0,
    poll_interval_seconds: float = 0.001,
) -> tuple[dict[str, object], dict[str, int]]:
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    start_event = context.Event()
    process = context.Process(
        target=_isolated_worker,
        args=(child_connection, start_event, operation, dict(payload)),
    )
    process.start()
    child_connection.close()

    baseline: int | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    deadline = time.monotonic() + baseline_timeout_seconds
    while baseline is None and time.monotonic() < deadline:
        message = _receive_message(parent_connection, 0.05)
        if message is not None:
            if message["stage"] == "baseline":
                baseline = int(message["rss"])
            elif message["stage"] == "error":
                error = str(message["traceback"])
                break
        if not process.is_alive():
            break

    if baseline is None:
        if process.is_alive():
            process.kill()
        process.join()
        parent_connection.close()
        detail = error or f"worker exited with code {process.exitcode}"
        raise RuntimeError(f"{operation} worker failed before RSS baseline:\n{detail}")

    monitored_process = psutil.Process(process.pid)
    peak = baseline
    start_event.set()
    while process.is_alive():
        try:
            peak = max(peak, int(monitored_process.memory_info().rss))
        except psutil.NoSuchProcess:
            pass
        while (message := _receive_message(parent_connection)) is not None:
            if message["stage"] == "result":
                result = dict(message["result"])
            elif message["stage"] == "error":
                error = str(message["traceback"])
        time.sleep(poll_interval_seconds)

    process.join()
    while (message := _receive_message(parent_connection)) is not None:
        if message["stage"] == "result":
            result = dict(message["result"])
        elif message["stage"] == "error":
            error = str(message["traceback"])
    parent_connection.close()
    exit_code = process.exitcode
    process.close()

    if error is not None or result is None or exit_code not in (0, None):
        detail = error or f"worker exited with code {exit_code}"
        raise RuntimeError(f"{operation} worker failed:\n{detail}")
    return result, {
        "rss_baseline_bytes": baseline,
        "peak_rss_bytes": peak,
        "peak_rss_delta_bytes": peak - baseline,
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_comparison_rows(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
) -> None:
    from experiments.problem_builders import generate_random_problems

    problems = generate_random_problems(_random_config(config))
    if len(rows) != len(problems):
        raise ValueError(
            f"comparison CSV has {len(rows)} rows but generator returned "
            f"{len(problems)} problems"
        )
    for expected_instance, (row, problem) in enumerate(zip(rows, problems)):
        expected = {
            "instance": expected_instance,
            "seed": int(problem["seed"]),
            "generation_attempt": int(problem["generation_attempt"]),
            "dimension": int(problem["matrix"].shape[0]),  # type: ignore[union-attr]
        }
        observed = {key: int(row[key]) for key in expected}
        if observed != expected:
            raise ValueError(
                "comparison CSV does not match regenerated problem at row "
                f"{expected_instance}: observed={observed}, expected={expected}"
            )


def add_comparison_memory_measurements(
    comparison_csv: Path = DEFAULT_COMPARISON_CSV,
    config_path: Path = DEFAULT_CONFIG,
) -> list[dict[str, str]]:
    fieldnames, rows = _read_csv(comparison_csv)
    config = _load_config(config_path)
    _validate_comparison_rows(rows, config)

    for row in rows:
        if row.get("speedup_core_valid") != "False" or row.get("speedup_core") != "":
            raise ValueError("speedup_core must remain invalid and empty")
        payload = {
            "config_path": str(config_path.resolve()),
            "instance": int(row["instance"]),
            "mu": int(row["mu"]),
            "tau": float(row["tau"]),
            "c_phys": float(row["C_phys"]),
        }
        _, qiskit_memory = _run_isolated("qiskit_exact", payload)
        _, tn_memory = _run_isolated("tn", payload)
        row.update(
            {
                "qiskit_exact_rss_baseline_bytes": str(
                    qiskit_memory["rss_baseline_bytes"]
                ),
                "qiskit_exact_peak_rss_bytes": str(
                    qiskit_memory["peak_rss_bytes"]
                ),
                "qiskit_exact_peak_rss_delta_bytes": str(
                    qiskit_memory["peak_rss_delta_bytes"]
                ),
                "tn_rss_baseline_bytes": str(tn_memory["rss_baseline_bytes"]),
                "tn_peak_rss_bytes": str(tn_memory["peak_rss_bytes"]),
                "tn_peak_rss_delta_bytes": str(
                    tn_memory["peak_rss_delta_bytes"]
                ),
            }
        )

    output_fields = fieldnames + [
        column for column in MEMORY_COLUMNS if column not in fieldnames
    ]
    _write_csv(comparison_csv, output_fields, rows)
    return rows


def run_scaling_experiment(
    *,
    n_values: Sequence[int] = (16, 32, 64, 128),
    mu_values: Sequence[int] = (16, 32, 64, 128, 256),
    tau: float = 20.0,
    repetitions: int = 5,
    seed: int = 12345,
    n_sweep_mu: int = 64,
    mu_sweep_n: int = 32,
) -> list[dict[str, object]]:
    points = [
        {"sweep": "N", "N": int(dimension), "mu": n_sweep_mu}
        for dimension in n_values
    ]
    points.extend(
        {"sweep": "mu", "N": mu_sweep_n, "mu": int(mu)} for mu in mu_values
    )
    rows: list[dict[str, object]] = []
    for point in points:
        payload = {
            **point,
            "tau": tau,
            "repetitions": repetitions,
            "seed": seed,
        }
        result, memory = _run_isolated("scaling", payload)
        result.update(memory)
        rows.append(result)
    return rows


def _summary_statistics(
    *,
    section: str,
    metric: str,
    method: str,
    values: Sequence[float],
) -> dict[str, object]:
    from experiments.run_reviewer_r1_c5_c6 import mean_statistics

    sample = np.asarray(values, dtype=np.float64)
    mean, sample_std, _, ci_low, ci_high = mean_statistics(sample)
    return {
        "section": section,
        "metric": metric,
        "method": method,
        "sample_count": sample.size,
        "mean": mean,
        "sample_std": sample_std,
        "median": float(np.median(sample)),
        "q1": float(np.percentile(sample, 25.0)),
        "q3": float(np.percentile(sample, 75.0)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def build_summary_rows(
    comparison_rows: Sequence[Mapping[str, str]],
    scaling_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    metrics = (
        ("fidelity_tn_qiskit", "TN versus exact Qiskit"),
        (
            "probability_rmse_tn_qiskit_exact",
            "TN versus exact Qiskit probabilities",
        ),
        ("probability_rmse_tn_direct", "TN versus direct solve"),
        ("probability_rmse_qiskit_direct", "exact Qiskit versus direct solve"),
        (
            "probability_rmse_sampled_exact",
            "sampled Qiskit versus exact Qiskit",
        ),
        ("success_probability_exact", "exact Qiskit"),
        ("success_probability_tn", "TN"),
        ("success_probability_sampled", "sampled Qiskit"),
        ("qiskit_total_exact_seconds", "exact Qiskit"),
        ("tn_total_seconds", "TN"),
        ("qiskit_exact_peak_rss_delta_bytes", "exact Qiskit"),
        ("tn_peak_rss_delta_bytes", "TN"),
    )
    summary = [
        _summary_statistics(
            section="qiskit_tn_comparison",
            metric=metric,
            method=method,
            values=[float(row[metric]) for row in comparison_rows],
        )
        for metric, method in metrics
    ]
    summary.append(
        _summary_statistics(
            section="qiskit_tn_comparison",
            metric="qiskit_total_exact_seconds_over_tn_total_seconds",
            method="per-instance exact Qiskit / TN",
            values=[
                float(row["qiskit_total_exact_seconds"])
                / float(row["tn_total_seconds"])
                for row in comparison_rows
            ],
        )
    )

    target_values = [
        1.0 if row["parameter_target_met"] == "True" else 0.0
        for row in comparison_rows
    ]
    target_count = int(sum(target_values))
    if target_count != 7 or len(target_values) != 20:
        raise ValueError(
            f"expected exactly 7/20 parameter targets met, got "
            f"{target_count}/{len(target_values)}"
        )
    summary.append(
        _summary_statistics(
            section="qiskit_tn_comparison",
            metric="parameter_target_met_fraction",
            method=f"selected pairs ({target_count}/{len(target_values)})",
            values=target_values,
        )
    )

    slope_specs = (
        ("N", "N", "tn_total_median_seconds", "loglog_slope_tn_time_vs_N"),
        (
            "N",
            "N",
            "peak_rss_delta_bytes",
            "loglog_slope_peak_rss_delta_vs_N",
        ),
        ("mu", "mu", "tn_total_median_seconds", "loglog_slope_tn_time_vs_mu"),
        (
            "mu",
            "mu",
            "peak_rss_delta_bytes",
            "loglog_slope_peak_rss_delta_vs_mu",
        ),
    )
    for sweep, x_key, y_key, metric in slope_specs:
        selected = [row for row in scaling_rows if row["sweep"] == sweep]
        x_values = np.asarray([float(row[x_key]) for row in selected])
        y_values = np.asarray([float(row[y_key]) for row in selected])
        positive = y_values > 0.0
        if np.count_nonzero(positive) < 2:
            raise ValueError(f"cannot fit {metric}: fewer than two positive values")
        slope = float(
            np.polyfit(np.log(x_values[positive]), np.log(y_values[positive]), 1)[0]
        )
        summary.append(
            {
                "section": "scaling",
                "metric": metric,
                "method": "empirical least-squares log-log fit",
                "sample_count": int(np.count_nonzero(positive)),
                "mean": slope,
                "sample_std": "",
                "median": "",
                "q1": "",
                "q3": "",
                "ci95_low": "",
                "ci95_high": "",
            }
        )
    return summary


def _plot_scaling(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    panels = (
        ("N", "N", "tn_total_median_seconds", "TN time vs N", True),
        ("N", "N", "peak_rss_delta_bytes", "Peak RSS delta vs N", False),
        ("mu", "mu", "tn_total_median_seconds", "TN time vs mu", True),
        ("mu", "mu", "peak_rss_delta_bytes", "Peak RSS delta vs mu", False),
    )
    for axis, (sweep, x_key, y_key, title, with_iqr) in zip(axes.flat, panels):
        selected = [row for row in rows if row["sweep"] == sweep]
        x_values = np.asarray([float(row[x_key]) for row in selected])
        y_values = np.asarray([float(row[y_key]) for row in selected])
        if with_iqr:
            q1 = np.asarray([float(row["tn_total_q1_seconds"]) for row in selected])
            q3 = np.asarray([float(row["tn_total_q3_seconds"]) for row in selected])
            axis.errorbar(
                x_values,
                y_values,
                yerr=np.vstack((y_values - q1, q3 - y_values)),
                marker="o",
                capsize=3,
            )
        else:
            positive = y_values > 0.0
            axis.plot(x_values[positive], y_values[positive], marker="o")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(
            xlabel=x_key,
            ylabel="seconds" if with_iqr else "bytes",
            title=title,
        )
        axis.grid(alpha=0.25, which="both")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={
            "Creator": "Tensor_Networks_HHL_algorithm",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def run_experiments(
    *,
    config_path: Path = DEFAULT_CONFIG,
    comparison_csv: Path = DEFAULT_COMPARISON_CSV,
    output_dir: Path = DEFAULT_OUTPUT,
) -> None:
    comparison_rows = add_comparison_memory_measurements(
        comparison_csv,
        config_path,
    )
    scaling_rows = run_scaling_experiment()
    summary_rows = build_summary_rows(comparison_rows, scaling_rows)
    scaling_fields = list(scaling_rows[0].keys())
    summary_fields = list(summary_rows[0].keys())
    _write_csv(output_dir / "scaling.csv", scaling_fields, scaling_rows)
    _write_csv(output_dir / "summary.csv", summary_fields, summary_rows)
    _plot_scaling(scaling_rows, output_dir / "scaling.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_experiments(
        config_path=args.config,
        comparison_csv=args.comparison_csv,
        output_dir=args.output_dir,
    )
    print(f"Generated Referee 2 C3/C4 results in {args.output_dir}")


if __name__ == "__main__":
    main()
