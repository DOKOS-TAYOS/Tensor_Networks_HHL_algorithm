from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from experiments.run_reviewer_r1_c5_c6 import (
    compare_tn_and_filter,
    random_instance_summary,
    run_experiments,
)
from experiments.run_reviewer_r2_c3_c4 import (
    DEFAULT_MU_SWEEP_N,
    DEFAULT_MU_VALUES,
    DEFAULT_N_SWEEP_MU,
    DEFAULT_N_VALUES,
    DEFAULT_TAU,
    MEMORY_COLUMNS,
    build_summary_rows,
    dominant_tensor_storage_estimate_bytes,
    run_scaling_experiment,
)
from experiments.run_reviewer_r2_c3_c4 import (
    run_experiments as run_reviewer_r2_experiments,
)

SCIENTIFIC_ARTIFACTS = {
    "application_results.csv",
    "spectral_diagnostics.csv",
    "tau_lambda_values.csv",
    "parameter_search.csv",
    "random_instance_results.csv",
    "random_instance_summary.csv",
    "hyperparameter_sweep.csv",
    "tn_filter_validation.csv",
    "OAF.pdf",
    "OAA.pdf",
    "C2D.pdf",
    "C2D_2D.pdf",
    "rmse_vs_tau.pdf",
    "rmse_vs_num_anc.pdf",
}


def test_runner_writes_only_scientific_results(tmp_path: Path) -> None:
    """Catch reintroduction of manifests, duplicate summaries, or PNG copies."""
    config = json.loads(
        Path("experiments/config_r1_c5_c6.json").read_text(encoding="utf-8")
    )
    config["mu_candidates"] = [32, 64, 128]
    config["tau_points_per_mu"] = 5
    config["target_rhs_filter_relative_error"] = 1.0
    config["timing_repetitions"] = 1
    config["random"]["instances"] = 2
    config["problems"]["harmonic_oscillator"]["T"] = 2.0
    config["problems"]["damped_oscillator"]["T"] = 2.0
    config["problems"]["heat_2d"]["nx"] = 4
    config["problems"]["heat_2d"]["ny"] = 4
    config["hyperparameter_sweep"]["tau_values"] = [1.0]
    config["hyperparameter_sweep"]["n_c_values"] = [5]
    config["tn_filter_validation"]["instance_ids"] = [0]
    config["tn_filter_validation"]["points"] = [{"n_c": 5, "tau": 1.0}]

    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "results"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_experiments(config_path, output_dir)

    generated = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert generated == SCIENTIFIC_ARTIFACTS

    with (output_dir / "random_instance_results.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        random_rows = list(csv.DictReader(stream))
    assert random_rows[0]["seed"] == "12345"
    assert int(random_rows[0]["generation_attempt"]) >= 1


def test_full_tn_matches_the_spectral_filter_on_an_exact_grid() -> None:
    problem = {
        "name": "exact_grid",
        "matrix": torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64)),
        "rhs": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "matrix_role": "test_hermitian_2x2",
    }

    row = compare_tn_and_filter(problem=problem, instance_id=0, n_c=3, tau=2.0)

    assert row["tn_filter_relative_difference"] < 1e-10
    assert row["tn_filter_rmse"] < 1e-10


def test_random_instance_summary_has_student_intervals() -> None:
    rows = [
        {
            "condition_number": 2.0,
            "rmse": 1.0,
            "relative_solution_error": 0.25,
            "normalized_residual": 0.125,
            "predicted_rhs_filter_relative_error": 0.0625,
        },
        {
            "condition_number": 4.0,
            "rmse": 3.0,
            "relative_solution_error": 0.75,
            "normalized_residual": 0.375,
            "predicted_rhs_filter_relative_error": 0.1875,
        },
    ]

    summary = random_instance_summary(rows)

    expected_columns = {
        "n_instances",
        "confidence_level",
        *{
            f"{metric}_{statistic}"
            for metric in (
                "condition_number",
                "rmse",
                "relative_solution_error",
                "normalized_residual",
                "predicted_rhs_filter_relative_error",
            )
            for statistic in ("mean", "std", "ci95_low", "ci95_high")
        },
    }
    assert set(summary) == expected_columns
    assert summary["n_instances"] == 2
    assert summary["confidence_level"] == 0.95

    sample = np.array([1.0, 3.0])
    half_width = stats.t.ppf(0.975, df=1) * np.std(sample, ddof=1) / np.sqrt(2)
    assert summary["rmse_mean"] == pytest.approx(np.mean(sample))
    assert summary["rmse_std"] == pytest.approx(np.std(sample, ddof=1))
    assert summary["rmse_ci95_low"] == pytest.approx(np.mean(sample) - half_width)
    assert summary["rmse_ci95_high"] == pytest.approx(np.mean(sample) + half_width)


def test_reviewer_r2_scaling_smoke() -> None:
    rows = run_scaling_experiment(
        n_values=(4, 8),
        mu_values=(8, 16),
        tau=10.0,
        repetitions=1,
        memory_repetitions=1,
        seed=12345,
        n_sweep_mu=16,
        mu_sweep_n=8,
    )

    assert rows
    assert len(rows) == 4
    for row in rows:
        baseline = float(row["tn_rss_baseline_median_bytes"])
        peak = float(row["tn_peak_rss_median_bytes"])
        delta = float(row["tn_peak_rss_delta_median_bytes"])
        assert baseline >= 0
        assert peak >= baseline
        assert delta == peak - baseline
        assert int(row["dominant_tensor_storage_estimate_bytes"]) > 0
        assert row["no_aliasing"] is True
        assert float(row["tn_filter_relative_difference"]) < 1e-9


def test_reviewer_r2_default_ranges_and_storage_estimate() -> None:
    assert DEFAULT_N_VALUES == (16, 32, 64, 96, 128, 192, 256)
    assert DEFAULT_MU_VALUES == (16, 32, 64, 128, 256, 512, 1024, 2048)
    assert DEFAULT_N_SWEEP_MU == 64
    assert DEFAULT_MU_SWEEP_N == 32
    assert DEFAULT_TAU == 20.0
    assert 20.0 * 0.35 < min(DEFAULT_MU_VALUES) / 2.0

    fixed_mu = [dominant_tensor_storage_estimate_bytes(n, 64) for n in (16, 32, 64)]
    fixed_n = [dominant_tensor_storage_estimate_bytes(32, mu) for mu in (16, 32, 64)]
    assert fixed_mu == sorted(fixed_mu)
    assert len(set(fixed_mu)) == len(fixed_mu)
    assert fixed_n == sorted(fixed_n)
    assert len(set(fixed_n)) == len(fixed_n)


def _comparison_rows_with_memory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        baseline = 100_000_000 + index
        for prefix, delta in (("qiskit_exact", 20_000_000), ("tn", 2_000_000)):
            row[f"{prefix}_rss_baseline_bytes"] = str(baseline)
            row[f"{prefix}_peak_rss_bytes"] = str(baseline + delta)
            row[f"{prefix}_peak_rss_delta_bytes"] = str(delta)
    return rows


def test_reviewer_r2_small_artifacts_regenerate(tmp_path: Path) -> None:
    comparison_rows = _comparison_rows_with_memory(
        Path("artifacts/reviewer_r1_c7_qiskit_comparison.csv")
    )
    comparison_csv = tmp_path / "comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    output_dir = tmp_path / "reviewer_r2"
    run_reviewer_r2_experiments(
        comparison_csv=comparison_csv,
        output_dir=output_dir,
        n_values=(4, 8, 12),
        mu_values=(8, 16, 32),
        tau=10.0,
        repetitions=1,
        memory_repetitions=1,
        n_sweep_mu=32,
        mu_sweep_n=8,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "scaling.csv",
        "summary.csv",
        "scaling.pdf",
    }
    with (output_dir / "scaling.csv").open(newline="", encoding="utf-8") as stream:
        scaling_rows = list(csv.DictReader(stream))
    assert len(scaling_rows) == 6
    assert "dominant_tensor_storage_estimate_bytes" in scaling_rows[0]
    assert "tn_peak_rss_delta_median_bytes" in scaling_rows[0]
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    metrics = {row["metric"] for row in summary_rows}
    assert "loglog_slope_tn_peak_rss_delta_vs_N" in metrics
    assert "loglog_slope_tn_peak_rss_delta_vs_mu" in metrics
    assert "loglog_slope_tn_peak_rss_vs_N" not in metrics


def test_loglog_memory_fit_ignores_nonpositive_deltas() -> None:
    comparison_rows = _comparison_rows_with_memory(
        Path("artifacts/reviewer_r1_c7_qiskit_comparison.csv")
    )
    scaling_rows: list[dict[str, object]] = []
    for sweep, x_key, deltas in (
        ("N", "N", (1.0, 0.0, -1.0)),
        ("mu", "mu", (1.0, 2.0, 4.0)),
    ):
        for index, delta in enumerate(deltas, start=1):
            scaling_rows.append(
                {
                    "sweep": sweep,
                    x_key: 2**index,
                    "tn_total_median_seconds": float(index),
                    "tn_peak_rss_delta_median_bytes": delta,
                }
            )

    summary = build_summary_rows(comparison_rows, scaling_rows)
    by_metric = {str(row["metric"]): row for row in summary}
    unavailable = by_metric["loglog_slope_tn_peak_rss_delta_vs_N"]
    assert unavailable["sample_count"] == 1
    assert unavailable["mean"] == ""
    assert "fewer than three" in str(unavailable["method"])
    fitted = by_metric["loglog_slope_tn_peak_rss_delta_vs_mu"]
    assert fitted["sample_count"] == 3
    np.testing.assert_allclose(float(fitted["mean"]), 1.0, atol=1e-12)
    assert not any("7/20" in str(value) for row in summary for value in row.values())
    assert set(MEMORY_COLUMNS) <= comparison_rows[0].keys()
