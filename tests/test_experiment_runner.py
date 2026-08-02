from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from experiments.run_reviewer_r1_c5_c6 import compare_tn_and_filter, run_experiments

SCIENTIFIC_ARTIFACTS = {
    "application_results.csv",
    "spectral_diagnostics.csv",
    "tau_lambda_values.csv",
    "parameter_search.csv",
    "random_instance_results.csv",
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
