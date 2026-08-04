from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, phase_estimation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from experiments.parameter_selection import evaluate_parameter_candidate
from experiments.run_reviewer_r1_c7 import select_reviewer_instances
from quantum_hhl import (
    HHL_circuit,
    ancilla_clock_zero_solution_state,
    ancilla_conditioned_density_matrix,
    build_hhl_unitary,
    controlled_rotation_angles,
    original_HHL_solver,
    run_qiskit_hhl_once,
    sampled_solution_probabilities,
)
from tn_hhl import tensornetwork_HHL


def test_exact_qiskit_run_can_skip_sampled_shots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_measurement(*args: object, **kwargs: object) -> QuantumCircuit:
        raise AssertionError("the measured circuit must not be constructed")

    monkeypatch.setattr("quantum_hhl.add_solution_measurements", reject_measurement)
    result = run_qiskit_hhl_once(
        n_ancillas=3,
        b_vector=torch.tensor([1.0, 2.0], dtype=torch.float64),
        A_matrix=torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)),
        tau=1.0,
        c_phys=0.25,
        n_shots=2_000,
        seed_transpiler=12345,
        seed_simulator=12345,
        include_shots=False,
    )

    assert result["shots_seconds"] == 0.0
    assert result["successful_ancilla_shots"] == 0
    assert result["sampled_ancilla_conditioned_probabilities"] is None
    assert result["sampled_ancilla_success_probability"] is None


def test_positive_and_negative_eigenvalues_reach_signed_qpe_bins() -> None:
    matrix = torch.diag(torch.tensor([1.0, -2.0], dtype=torch.float64))
    unitary = build_hhl_unitary(matrix, tau=1.0, mu=8)

    assert np.diag(unitary) == pytest.approx(
        [(1.0 + 1j) / np.sqrt(2.0), -1j], abs=1e-12
    )

    for eigenstate, expected_signed_bin in ((0, 1), (1, -2)):
        circuit = QuantumCircuit(4)
        if eigenstate == 1:
            circuit.x(3)
        circuit.append(phase_estimation(3, UnitaryGate(unitary)), range(4))
        statevector = np.asarray(Statevector.from_instruction(circuit))
        clock_probabilities = np.abs(statevector.reshape(2, 8)) ** 2
        physical_bin = int(np.argmax(clock_probabilities.sum(axis=0)))
        logical_bin = int(f"{physical_bin:03b}"[::-1], 2)
        signed_bin = logical_bin if logical_bin <= 4 else logical_bin - 8

        assert signed_bin == expected_signed_bin


def test_controlled_rotations_use_c_bin_and_reject_invalid_domain() -> None:
    angles, c_bin = controlled_rotation_angles(mu=8, tau=2.0, c_phys=0.25)

    assert c_bin == pytest.approx(0.5)
    assert np.sin(angles / 2.0) == pytest.approx(
        [0.0, 0.5, 0.25, 1.0 / 6.0, 0.125, -1.0 / 6.0, -0.25, -0.5],
        abs=1e-12,
    )

    with pytest.raises(ValueError, match="controlled-rotation domain"):
        controlled_rotation_angles(mu=8, tau=2.0, c_phys=0.51)


def test_postselection_extractors_respect_qiskit_bit_order() -> None:
    statevector = np.zeros(32, dtype=np.complex128)
    statevector[0] = np.sqrt(0.6)
    statevector[1 + 2 * 0 + 8 * 1] = np.sqrt(0.3)
    statevector[1 + 2 * 3 + 8 * 2] = 1j * np.sqrt(0.1)

    density_matrix, success_probability = ancilla_conditioned_density_matrix(
        statevector,
        n_ancillas=2,
        n_state_qubits=2,
    )
    clock_zero_state, joint_probability = ancilla_clock_zero_solution_state(
        statevector,
        n_ancillas=2,
        n_state_qubits=2,
    )

    assert success_probability == pytest.approx(0.4)
    assert np.diag(density_matrix).real == pytest.approx([0.0, 0.75, 0.25, 0.0])
    assert density_matrix[1, 2] == pytest.approx(0.0)
    assert np.trace(density_matrix) == pytest.approx(1.0)
    assert joint_probability == pytest.approx(0.3)
    assert clock_zero_state is not None
    np.testing.assert_allclose(clock_zero_state, [0.0, 1.0, 0.0, 0.0])


def test_sampled_postselection_respects_measurement_bit_order() -> None:
    probabilities, successful_shots, success_probability = (
        sampled_solution_probabilities(
            {"001": 10, "101": 20, "110": 70},
            n_state_qubits=2,
        )
    )

    assert successful_shots == 30
    assert success_probability == pytest.approx(0.3)
    assert probabilities == pytest.approx([1.0 / 3.0, 0.0, 2.0 / 3.0, 0.0])


@pytest.mark.parametrize(
    ("tau", "expected_c_phys", "expected_c_bin"),
    [(1.5, 0.45, 0.675), (3.0, 0.3, 0.9)],
)
def test_adaptive_rotation_constant_covers_both_regimes(
    tau: float,
    expected_c_phys: float,
    expected_c_bin: float,
) -> None:
    candidate = evaluate_parameter_candidate(
        problem="rotation_regime",
        eigenvalues=np.array([-1.0, 0.5]),
        eigenvectors=np.eye(2),
        rhs=np.array([1.0, 1.0]),
        mu=8,
        tau=tau,
        zero_bin_separation=0.1,
        target_rhs_filter_relative_error=1.0,
        controlled_rotation_safety_factor=0.9,
    )
    unconstrained = evaluate_parameter_candidate(
        problem="rotation_regime",
        eigenvalues=np.array([-1.0, 0.5]),
        eigenvectors=np.eye(2),
        rhs=np.array([1.0, 1.0]),
        mu=8,
        tau=tau,
        zero_bin_separation=0.1,
        target_rhs_filter_relative_error=1.0,
    )

    assert candidate["C_phys"] == pytest.approx(expected_c_phys)
    assert candidate["C_bin"] == pytest.approx(expected_c_bin)
    assert candidate["rotation_valid"]
    assert candidate["rhs_relative_filter_error"] == pytest.approx(
        unconstrained["rhs_relative_filter_error"]
    )
    assert candidate["target_met"] == unconstrained["target_met"]


@pytest.mark.parametrize("factor", [0.0, -0.1, 1.01, np.inf, np.nan])
def test_rotation_safety_factor_must_be_finite_and_bounded(factor: float) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        evaluate_parameter_candidate(
            problem="invalid_safety_factor",
            eigenvalues=np.array([-1.0, 0.5]),
            eigenvectors=np.eye(2),
            rhs=np.array([1.0, 1.0]),
            mu=8,
            tau=1.0,
            zero_bin_separation=0.1,
            target_rhs_filter_relative_error=1.0,
            controlled_rotation_safety_factor=factor,
        )


def test_parameter_candidate_rejects_half_bin_zero_tie() -> None:
    candidate = evaluate_parameter_candidate(
        problem="half_bin_tie",
        eigenvalues=np.array([-1.0, 0.5]),
        eigenvectors=np.eye(2),
        rhs=np.array([1.0, 1.0]),
        mu=8,
        tau=1.0,
        zero_bin_separation=0.5,
        target_rhs_filter_relative_error=1.0,
        controlled_rotation_safety_factor=0.9,
    )

    assert not candidate["feasible"]
    assert candidate["rejection_reason"] == "zero_bin_separation"


def test_tn_reports_the_requested_scientific_timing_phases() -> None:
    timings: dict[str, float] = {}
    result = tensornetwork_HHL(
        8,
        2.0,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64)),
        timings=timings,
    )

    assert torch.isfinite(result).all()
    assert set(timings) == {
        "unitary_seconds",
        "preparation_seconds",
        "contraction_seconds",
    }
    assert all(duration >= 0.0 for duration in timings.values())


def test_exact_grid_postselections_coincide() -> None:
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    matrix = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    result = run_qiskit_hhl_once(
        n_ancillas=3,
        b_vector=rhs,
        A_matrix=matrix,
        tau=1.0,
        c_phys=0.25,
        n_shots=2_000,
        seed_transpiler=12345,
        seed_simulator=12345,
    )

    density_matrix = result["ancilla_conditioned_density_matrix"]
    clock_zero_state = result["clock_zero_solution_state"]
    assert isinstance(density_matrix, np.ndarray)
    assert np.trace(density_matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(
        density_matrix,
        [[0.2, -0.4], [-0.4, 0.8]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        density_matrix,
        np.outer(clock_zero_state, np.asarray(clock_zero_state).conj()),
        atol=1e-12,
    )
    assert result["ancilla_success_probability"] == pytest.approx(0.25**2, abs=1e-12)
    assert result["ancilla_clock_zero_probability"] == pytest.approx(
        result["ancilla_success_probability"], abs=1e-12
    )
    assert (
        np.linalg.norm(result["sampled_ancilla_conditioned_probabilities"] - [0.2, 0.8])
        < 0.05
    )
    assert 0 < result["successful_ancilla_shots"] <= 2_000
    assert 0.0 <= result["sampled_ancilla_success_probability"] <= 1.0

    timing_keys = {
        "unitary_seconds",
        "circuit_seconds",
        "transpile_seconds",
        "statevector_seconds",
        "extraction_seconds",
        "shots_seconds",
        "total_exact_seconds",
    }
    assert timing_keys <= result.keys()
    assert all(float(result[key]) >= 0.0 for key in timing_keys)


def test_off_grid_joint_branch_matches_tn_but_ancilla_state_is_mixed() -> None:
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    matrix = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    tau = 1.37
    c_phys = 0.25
    result = run_qiskit_hhl_once(
        n_ancillas=3,
        b_vector=rhs,
        A_matrix=matrix,
        tau=tau,
        c_phys=c_phys,
        n_shots=100,
        seed_transpiler=12345,
        seed_simulator=12345,
        include_shots=False,
    )
    tn_vector = tensornetwork_HHL(8, tau, rhs, matrix).detach().cpu().numpy()
    tn_state = tn_vector / np.linalg.norm(tn_vector)
    density_matrix = np.asarray(result["ancilla_conditioned_density_matrix"])
    clock_zero_state = np.asarray(result["clock_zero_solution_state"])
    purity = float(np.trace(density_matrix @ density_matrix).real)
    tn_joint_probability = float(
        c_phys**2 * np.linalg.norm(tn_vector) ** 2 / np.linalg.norm(rhs.numpy()) ** 2
    )

    assert result["ancilla_success_probability"] != pytest.approx(
        result["ancilla_clock_zero_probability"], abs=1e-6
    )
    assert purity < 1.0 - 1e-6
    assert abs(np.vdot(tn_state, clock_zero_state)) ** 2 == pytest.approx(
        1.0, abs=1e-12
    )
    assert np.vdot(tn_state, density_matrix @ tn_state).real < 1.0 - 1e-6
    assert result["ancilla_clock_zero_probability"] == pytest.approx(
        tn_joint_probability, abs=1e-12
    )


def test_all_reviewer_instances_meet_the_one_percent_target() -> None:
    config = json.loads(
        Path("experiments/config_r1_c5_c6.json").read_text(encoding="utf-8")
    )
    selected_instances = select_reviewer_instances(config)

    assert len(selected_instances) == 20
    assert all(item["selected"]["target_met"] for item in selected_instances)
    assert (
        max(
            float(item["selected"]["rhs_relative_filter_error"])
            for item in selected_instances
        )
        <= 0.01
    )
    assert {int(item["selected"]["mu"]) for item in selected_instances} <= {
        128,
        256,
        512,
        1024,
        2048,
    }


def test_reviewer_comparison_csv_contains_twenty_valid_instances() -> None:
    csv_path = Path("artifacts/reviewer_r1_c7_qiskit_comparison.csv")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 20
    assert {int(row["instance"]) for row in rows} == set(range(20))
    scientific_columns = {
        "ancilla_success_probability",
        "ancilla_clock_zero_probability",
        "tn_clock_zero_equivalent_probability",
        "ancilla_clock_zero_probability_absolute_difference",
        "fidelity_tn_ancilla_clock_zero",
        "fidelity_tn_ancilla",
        "ancilla_conditioned_purity",
        "probability_rmse_tn_ancilla_clock_zero",
        "probability_rmse_tn_ancilla",
        "probability_rmse_sampled_ancilla_exact",
        "sampled_ancilla_success_probability",
        "qiskit_exact_rss_baseline_bytes",
        "qiskit_exact_peak_rss_bytes",
        "qiskit_exact_peak_rss_delta_bytes",
        "tn_rss_baseline_bytes",
        "tn_peak_rss_bytes",
        "tn_peak_rss_delta_bytes",
    }
    assert scientific_columns <= rows[0].keys()
    assert "success_probability_exact" not in rows[0]
    for row in rows:
        assert int(row["seed"]) == 12345
        assert int(row["dimension"]) == 16
        assert int(row["mu"]) in {128, 256, 512, 1024, 2048}
        assert row["no_aliasing"] == "True"
        assert row["singular_bin_assigned"] == "False"
        assert row["zero_bin_separated"] == "True"
        assert row["rotation_valid"] == "True"
        assert row["parameter_target_met"] == "True"
        assert float(row["predicted_rhs_filter_relative_error"]) <= 0.01
        for column in (
            "ancilla_success_probability",
            "ancilla_clock_zero_probability",
            "tn_clock_zero_equivalent_probability",
            "fidelity_tn_ancilla_clock_zero",
            "fidelity_tn_ancilla",
            "ancilla_conditioned_purity",
            "sampled_ancilla_success_probability",
        ):
            assert 0.0 <= float(row[column]) <= 1.0
        assert float(row["ancilla_clock_zero_probability_absolute_difference"]) <= 2e-9
        assert row["speedup_core_valid"] == "False"
        assert row["speedup_core"] == ""
        for prefix in ("qiskit_exact", "tn"):
            baseline = int(row[f"{prefix}_rss_baseline_bytes"])
            peak = int(row[f"{prefix}_peak_rss_bytes"])
            delta = int(row[f"{prefix}_peak_rss_delta_bytes"])
            assert baseline >= 0
            assert peak >= baseline
            assert delta == peak - baseline


def test_legacy_keywords_and_measurement_layout_remain_available() -> None:
    with pytest.warns(DeprecationWarning, match="interpreted as tau"):
        circuit = HHL_circuit(
            3,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)),
            t=1.0,
            C=0.25,
        )

    assert [register.name for register in circuit.cregs] == [
        "canc",
        "cClock",
        "cState",
    ]
    assert circuit.num_clbits == 1 + 3 + 1

    with pytest.warns(DeprecationWarning, match="interpreted as tau"):
        probabilities = original_HHL_solver(
            3,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)),
            t=1.0,
            C=0.25,
            n_shots=100,
            backend=AerSimulator(method="statevector"),
        )
    assert probabilities.sum().item() == pytest.approx(1.0)
