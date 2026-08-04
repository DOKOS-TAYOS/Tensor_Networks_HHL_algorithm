from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, phase_estimation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from experiments.parameter_selection import evaluate_parameter_candidate
from quantum_hhl import (
    HHL_circuit,
    build_hhl_unitary,
    conditioned_solution_density_matrix,
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
    assert result["successful_shots"] == 0
    assert result["sampled_probabilities"] is None
    assert result["success_probability_sampled"] is None


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


def test_conditioned_reduced_state_respects_qiskit_bit_order() -> None:
    statevector = np.zeros(32, dtype=np.complex128)
    statevector[0] = np.sqrt(0.6)
    statevector[1 + 2 * 0 + 8 * 1] = np.sqrt(0.3)
    statevector[1 + 2 * 3 + 8 * 2] = 1j * np.sqrt(0.1)

    density_matrix, success_probability = conditioned_solution_density_matrix(
        statevector,
        n_ancillas=2,
        n_state_qubits=2,
    )

    assert success_probability == pytest.approx(0.4)
    assert np.diag(density_matrix).real == pytest.approx([0.0, 0.75, 0.25, 0.0])
    assert density_matrix[1, 2] == pytest.approx(0.0)
    assert np.trace(density_matrix) == pytest.approx(1.0)


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


def test_parameter_candidate_rejects_invalid_rotation_without_clipping() -> None:
    eigenvalues = np.array([-1.0, 0.5])
    eigenvectors = np.eye(2)
    rhs = np.array([1.0, 1.0])

    candidate = evaluate_parameter_candidate(
        problem="rotation_domain",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        rhs=rhs,
        mu=8,
        tau=3.0,
        zero_bin_separation=0.5,
        target_rhs_filter_relative_error=1.0,
        controlled_rotation_scale=0.9,
    )

    assert candidate["C_phys"] == pytest.approx(0.45)
    assert candidate["C_bin"] == pytest.approx(1.35)
    assert not candidate["rotation_valid"]
    assert not candidate["feasible"]
    assert candidate["rejection_reason"] == "controlled_rotation_domain"


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
        controlled_rotation_scale=0.9,
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


def test_exact_qiskit_run_conditions_on_ancilla_and_traces_clock() -> None:
    result = run_qiskit_hhl_once(
        n_ancillas=3,
        b_vector=torch.tensor([1.0, 2.0], dtype=torch.float64),
        A_matrix=torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)),
        tau=1.0,
        c_phys=0.25,
        n_shots=2_000,
        seed_transpiler=12345,
        seed_simulator=12345,
    )

    density_matrix = result["density_matrix"]
    assert isinstance(density_matrix, np.ndarray)
    assert np.trace(density_matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(
        density_matrix,
        [[0.2, -0.4], [-0.4, 0.8]],
        atol=1e-12,
    )
    assert result["success_probability_exact"] == pytest.approx(0.25**2, abs=1e-12)
    assert np.linalg.norm(result["sampled_probabilities"] - [0.2, 0.8]) < 0.05
    assert 0 < result["successful_shots"] <= 2_000
    assert 0.0 <= result["success_probability_sampled"] <= 1.0

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


def test_reviewer_comparison_csv_contains_twenty_valid_instances() -> None:
    csv_path = Path("artifacts/reviewer_r1_c7_qiskit_comparison.csv")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 20
    assert {int(row["instance"]) for row in rows} == set(range(20))
    memory_columns = {
        "qiskit_exact_rss_baseline_bytes",
        "qiskit_exact_peak_rss_bytes",
        "qiskit_exact_peak_rss_delta_bytes",
        "tn_rss_baseline_bytes",
        "tn_peak_rss_bytes",
        "tn_peak_rss_delta_bytes",
    }
    assert memory_columns <= rows[0].keys()
    for row in rows:
        assert int(row["seed"]) == 12345
        assert int(row["dimension"]) == 16
        assert int(row["mu"]) in {128, 256, 512, 1024}
        assert row["no_aliasing"] == "True"
        assert row["singular_bin_assigned"] == "False"
        assert row["rotation_valid"] == "True"
        assert 0.0 <= float(row["success_probability_exact"]) <= 1.0
        assert 0.0 <= float(row["fidelity_tn_qiskit"]) <= 1.0
        assert row["speedup_core_valid"] == "False"
        assert row["speedup_core"] == ""
        assert all(int(row[column]) >= 0 for column in memory_columns)


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
