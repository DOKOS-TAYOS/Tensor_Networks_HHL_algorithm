from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
import warnings

import numpy as np
import torch
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import (
    StatePreparation,
    UCRYGate,
    UnitaryGate,
    phase_estimation,
)
from qiskit_aer import AerSimulator
from scipy.linalg import expm

from experiments.parameter_selection import signed_phase_indices


def _as_effectively_real_numpy(tensor: torch.Tensor, *, name: str) -> np.ndarray:
    """Convert a tensor to NumPy after rejecting a genuine imaginary part."""
    values = tensor.detach().cpu().numpy()
    if np.iscomplexobj(values):
        if np.max(np.abs(values.imag)) > 1e-12:
            raise ValueError(f"{name} must be effectively real")
        values = values.real
    return np.asarray(values, dtype=np.float64)


def _padded_hhl_problem(
    b_vector: torch.Tensor,
    A_matrix: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad one real linear system to the state-register dimension."""
    b_values = _as_effectively_real_numpy(b_vector, name="b_vector")
    matrix = _as_effectively_real_numpy(A_matrix, name="A_matrix")
    if b_values.ndim != 1:
        raise ValueError("b_vector must be one-dimensional")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A_matrix must be square")
    if matrix.shape[0] != b_values.size:
        raise ValueError("A_matrix and b_vector dimensions must match")
    if b_values.size < 2:
        raise ValueError("b_vector must contain at least two entries")

    n_state_qubits = int(np.ceil(np.log2(b_values.size)))
    padded_dimension = 2**n_state_qubits
    b_padded = np.pad(b_values, (0, padded_dimension - b_values.size))
    A_padded = np.eye(padded_dimension, dtype=np.float64)
    A_padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return b_padded, A_padded, n_state_qubits


def build_hhl_unitary(
    A_matrix: torch.Tensor,
    tau: float,
    mu: int,
) -> np.ndarray:
    """Build ``U = exp(2*pi*i*tau*A/mu)`` on the padded state space."""
    if mu < 2 or mu & (mu - 1):
        raise ValueError("mu must be a power of two")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    dummy_rhs = torch.zeros(A_matrix.shape[0], dtype=torch.float64)
    _, A_padded, _ = _padded_hhl_problem(dummy_rhs, A_matrix)
    phase_scale = 2.0 * np.pi * tau / mu
    return expm(1j * phase_scale * A_padded)


def controlled_rotation_angles(
    mu: int,
    tau: float,
    c_phys: float,
) -> tuple[np.ndarray, float]:
    """Return signed-bin HHL angles with ``C_bin = tau*C_phys``.

    The singular bin receives angle zero. No clipping is performed: an invalid
    ``arcsin`` argument raises before circuit construction.
    """
    if mu < 2 or mu & (mu - 1):
        raise ValueError("mu must be a power of two")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    if not np.isfinite(c_phys) or c_phys <= 0.0:
        raise ValueError("C_phys must be finite and positive")

    c_bin = tau * c_phys
    signed_bins = signed_phase_indices(mu)
    nonzero = signed_bins != 0
    ratios = np.zeros(mu, dtype=np.float64)
    ratios[nonzero] = c_bin / signed_bins[nonzero]
    invalid = nonzero & (np.abs(ratios) > 1.0)
    if np.any(invalid):
        invalid_bins = np.flatnonzero(invalid).tolist()
        raise ValueError(
            "controlled-rotation domain is invalid for phase bins "
            f"{invalid_bins}: C_bin={c_bin:.16g}"
        )
    return 2.0 * np.arcsin(ratios), float(c_bin)


def add_solution_measurements(
    circuit: QuantumCircuit,
    n_ancillas: int,
    n_state_qubits: int,
) -> QuantumCircuit:
    """Measure solution qubits and the success ancilla in a parseable order."""
    measured = circuit.copy()
    classical = ClassicalRegister(n_state_qubits + 1, "solution_success")
    measured.add_register(classical)
    measured.measure(measured.qubits[0], classical[0])
    state_offset = 1 + n_ancillas
    for state_qubit in range(n_state_qubits):
        measured.measure(
            measured.qubits[state_offset + state_qubit],
            classical[state_qubit + 1],
        )
    return measured


def _add_legacy_measurements(
    circuit: QuantumCircuit,
    n_ancillas: int,
    n_state_qubits: int,
) -> QuantumCircuit:
    """Preserve the historical ancilla, clock, and state register layout."""
    measured = circuit.copy()
    ancilla_bits = ClassicalRegister(1, "canc")
    clock_bits = ClassicalRegister(n_ancillas, "cClock")
    state_bits = ClassicalRegister(n_state_qubits, "cState")
    measured.add_register(ancilla_bits, clock_bits, state_bits)
    measured.measure(measured.qubits[0], ancilla_bits[0])
    for clock_qubit in range(n_ancillas):
        measured.measure(measured.qubits[1 + clock_qubit], clock_bits[clock_qubit])
    state_offset = 1 + n_ancillas
    for state_qubit in range(n_state_qubits):
        measured.measure(
            measured.qubits[state_offset + state_qubit],
            state_bits[state_qubit],
        )
    return measured


def HHL_circuit(
    n_ancillas: int,
    b_vector: torch.Tensor,
    A_matrix: torch.Tensor,
    tau: float | None = None,
    c_phys: float | None = None,
    *,
    U_matrix: np.ndarray | None = None,
    measure: bool = True,
    **legacy_parameters: float,
) -> QuantumCircuit:
    """Construct the finite-resolution HHL circuit for a common ``(mu, tau)``."""
    used_legacy_names = "t" in legacy_parameters or "C" in legacy_parameters
    if "t" in legacy_parameters:
        if tau is not None:
            raise TypeError("specify tau or deprecated t, not both")
        tau = legacy_parameters.pop("t")
    if "C" in legacy_parameters:
        if c_phys is not None:
            raise TypeError("specify c_phys or deprecated C, not both")
        c_phys = legacy_parameters.pop("C")
    if legacy_parameters:
        unexpected = ", ".join(sorted(legacy_parameters))
        raise TypeError(f"unexpected HHL parameters: {unexpected}")
    if tau is None or c_phys is None:
        raise TypeError("tau and c_phys are required")
    if used_legacy_names:
        warnings.warn(
            "Deprecated t is interpreted as tau (the evolution coefficient is "
            "2*pi*tau/mu), and deprecated C is interpreted as C_phys.",
            DeprecationWarning,
            stacklevel=2,
        )
    mu = 2**n_ancillas
    b_padded, _, n_state_qubits = _padded_hhl_problem(b_vector, A_matrix)
    evolution = (
        build_hhl_unitary(A_matrix, tau, mu)
        if U_matrix is None
        else np.asarray(U_matrix, dtype=np.complex128)
    )
    expected_shape = (2**n_state_qubits, 2**n_state_qubits)
    if evolution.shape != expected_shape:
        raise ValueError(f"U_matrix must have shape {expected_shape}")
    angles, _ = controlled_rotation_angles(mu, tau, c_phys)

    ancilla_reg = QuantumRegister(1, "Anc")
    clock_reg = QuantumRegister(n_ancillas, "Clock")
    state_reg = QuantumRegister(n_state_qubits, "State")
    circuit = QuantumCircuit(ancilla_reg, clock_reg, state_reg, name="HHL")
    circuit.append(StatePreparation(b_padded, normalize=True), state_reg)

    qpe = phase_estimation(n_ancillas, UnitaryGate(evolution, label="U"))
    qpe_qubits = clock_reg[:] + state_reg[:]
    circuit.append(qpe, qpe_qubits)
    # Qiskit's QPE presents the logical phase bits in the reverse physical
    # qubit order expected by UCRYGate's little-endian control-state index.
    circuit.append(
        UCRYGate(angles.tolist()),
        ancilla_reg[:] + list(reversed(clock_reg[:])),
    )
    circuit.append(qpe.inverse(), qpe_qubits)
    circuit = circuit.decompose(["QPE", "QPE_dg"], reps=2)

    if measure:
        return _add_legacy_measurements(circuit, n_ancillas, n_state_qubits)
    return circuit


def ancilla_conditioned_density_matrix(
    statevector: np.ndarray,
    n_ancillas: int,
    n_state_qubits: int,
) -> tuple[np.ndarray, float]:
    """Condition on ancilla ``1`` and trace out Qiskit's phase register."""
    amplitudes = np.asarray(statevector, dtype=np.complex128)
    expected_size = 2 ** (1 + n_ancillas + n_state_qubits)
    if amplitudes.shape != (expected_size,):
        raise ValueError(f"statevector must have length {expected_size}")

    # Qiskit uses little-endian qubit indices: Anc is the least-significant
    # bit, followed by Clock, while State contains the most-significant bits.
    ordered = amplitudes.reshape(2**n_state_qubits, 2**n_ancillas, 2)
    ancilla_branch = ordered[:, :, 1]
    ancilla_success_probability = float(np.sum(np.abs(ancilla_branch) ** 2))
    if not 0.0 < ancilla_success_probability <= 1.0 + 1e-12:
        raise ValueError("ancilla success probability must lie in (0, 1]")

    density_matrix = (
        ancilla_branch @ ancilla_branch.conj().T / ancilla_success_probability
    )
    return density_matrix, min(ancilla_success_probability, 1.0)


def ancilla_clock_zero_solution_state(
    statevector: np.ndarray,
    n_ancillas: int,
    n_state_qubits: int,
) -> tuple[np.ndarray | None, float]:
    """Extract the normalized ``ancilla=1, clock=0`` Qiskit branch."""
    amplitudes = np.asarray(statevector, dtype=np.complex128)
    expected_size = 2 ** (1 + n_ancillas + n_state_qubits)
    if amplitudes.shape != (expected_size,):
        raise ValueError(f"statevector must have length {expected_size}")

    # Anc is Qiskit's least-significant bit, followed by Clock; therefore the
    # joint branch is ordered[:, 0, 1] after grouping State, Clock, Anc.
    ordered = amplitudes.reshape(2**n_state_qubits, 2**n_ancillas, 2)
    branch = ordered[:, 0, 1].copy()
    probability = float(np.vdot(branch, branch).real)
    if not -1e-12 <= probability <= 1.0 + 1e-12:
        raise ValueError("ancilla-clock-zero probability must lie in [0, 1]")
    probability = min(max(probability, 0.0), 1.0)
    if probability == 0.0:
        return None, probability
    return branch / np.sqrt(probability), probability


def sampled_solution_probabilities(
    counts: Mapping[str, int],
    n_state_qubits: int,
) -> tuple[np.ndarray, int, float]:
    """Postselect sampled counts encoded as ``state_bits + ancilla_bit``."""
    probabilities = np.zeros(2**n_state_qubits, dtype=np.float64)
    successful_shots = 0
    total_shots = 0
    for raw_key, count in counts.items():
        key = raw_key.replace(" ", "")
        if len(key) != n_state_qubits + 1:
            raise ValueError("unexpected measurement key width")
        total_shots += int(count)
        if key[-1] != "1":
            continue
        successful_shots += int(count)
        probabilities[int(key[:-1], 2)] += int(count)

    if total_shots <= 0:
        raise ValueError("shot counts must be non-empty")
    if successful_shots <= 0:
        raise ValueError("no successful ancilla shots were observed")
    probabilities /= successful_shots
    return probabilities, successful_shots, successful_shots / total_shots


def run_qiskit_hhl_once(
    *,
    n_ancillas: int,
    b_vector: torch.Tensor,
    A_matrix: torch.Tensor,
    tau: float,
    c_phys: float,
    n_shots: int,
    seed_transpiler: int,
    seed_simulator: int,
    threads: int = 1,
    include_shots: bool = True,
) -> dict[str, object]:
    """Run one timed exact simulation and its secondary sampled comparison."""
    if n_shots < 1:
        raise ValueError("n_shots must be positive")
    if threads < 1:
        raise ValueError("threads must be positive")
    mu = 2**n_ancillas
    n_state_qubits = int(np.ceil(np.log2(len(b_vector))))
    backend = AerSimulator(
        method="statevector",
        precision="double",
        max_parallel_threads=threads,
    )

    start = perf_counter()
    unitary = build_hhl_unitary(A_matrix, tau, mu)
    unitary_seconds = perf_counter() - start

    start = perf_counter()
    circuit = HHL_circuit(
        n_ancillas,
        b_vector,
        A_matrix,
        tau,
        c_phys,
        U_matrix=unitary,
        measure=False,
    )
    circuit_seconds = perf_counter() - start

    start = perf_counter()
    transpiled = transpile(
        circuit,
        backend,
        seed_transpiler=seed_transpiler,
    )
    transpile_seconds = perf_counter() - start

    exact_circuit = transpiled.copy()
    exact_circuit.save_statevector()
    start = perf_counter()
    exact_result = backend.run(
        exact_circuit,
        seed_simulator=seed_simulator,
    ).result()
    statevector = np.asarray(exact_result.get_statevector(exact_circuit))
    statevector_seconds = perf_counter() - start

    start = perf_counter()
    ancilla_density_matrix, ancilla_success_probability = (
        ancilla_conditioned_density_matrix(
            statevector,
            n_ancillas,
            n_state_qubits,
        )
    )
    clock_zero_state, ancilla_clock_zero_probability = (
        ancilla_clock_zero_solution_state(
            statevector,
            n_ancillas,
            n_state_qubits,
        )
    )
    if clock_zero_state is None:
        raise ValueError("the ancilla=1, clock=0 branch has zero probability")
    ancilla_conditioned_probabilities = np.diag(ancilla_density_matrix).real.copy()
    ancilla_clock_zero_probabilities = np.abs(clock_zero_state) ** 2
    extraction_seconds = perf_counter() - start

    sampled_ancilla_conditioned_probabilities: np.ndarray | None = None
    successful_ancilla_shots = 0
    sampled_ancilla_success_probability: float | None = None
    shots_seconds = 0.0
    if include_shots:
        measured_circuit = add_solution_measurements(
            transpiled,
            n_ancillas,
            n_state_qubits,
        )
        start = perf_counter()
        counts = backend.run(
            measured_circuit,
            shots=n_shots,
            seed_simulator=seed_simulator,
        ).result().get_counts()
        shots_seconds = perf_counter() - start
        (
            sampled_ancilla_conditioned_probabilities,
            successful_ancilla_shots,
            sampled_ancilla_success_probability,
        ) = sampled_solution_probabilities(counts, n_state_qubits)

    total_exact_seconds = (
        unitary_seconds
        + circuit_seconds
        + transpile_seconds
        + statevector_seconds
        + extraction_seconds
    )
    return {
        "ancilla_conditioned_density_matrix": ancilla_density_matrix,
        "ancilla_conditioned_probabilities": ancilla_conditioned_probabilities,
        "ancilla_success_probability": ancilla_success_probability,
        "clock_zero_solution_state": clock_zero_state,
        "ancilla_clock_zero_probabilities": ancilla_clock_zero_probabilities,
        "ancilla_clock_zero_probability": ancilla_clock_zero_probability,
        "sampled_ancilla_conditioned_probabilities": (
            sampled_ancilla_conditioned_probabilities
        ),
        "sampled_ancilla_success_probability": (
            sampled_ancilla_success_probability
        ),
        "successful_ancilla_shots": successful_ancilla_shots,
        "unitary_seconds": unitary_seconds,
        "circuit_seconds": circuit_seconds,
        "transpile_seconds": transpile_seconds,
        "statevector_seconds": statevector_seconds,
        "extraction_seconds": extraction_seconds,
        "shots_seconds": shots_seconds,
        "total_exact_seconds": total_exact_seconds,
    }


def original_HHL_solver(
    n_ancillas: int,
    b_vector: torch.Tensor,
    A_matrix: torch.Tensor,
    tau: float | None = None,
    c_phys: float | None = None,
    n_shots: int | None = None,
    backend: AerSimulator | None = None,
    *,
    seed_transpiler: int = 12345,
    seed_simulator: int = 12345,
    **legacy_parameters: float,
) -> torch.Tensor:
    """Execute the secondary sampled HHL comparison with fixed seeds."""
    used_legacy_names = "t" in legacy_parameters or "C" in legacy_parameters
    if "t" in legacy_parameters:
        if tau is not None:
            raise TypeError("specify tau or deprecated t, not both")
        tau = legacy_parameters.pop("t")
    if "C" in legacy_parameters:
        if c_phys is not None:
            raise TypeError("specify c_phys or deprecated C, not both")
        c_phys = legacy_parameters.pop("C")
    if legacy_parameters:
        unexpected = ", ".join(sorted(legacy_parameters))
        raise TypeError(f"unexpected HHL parameters: {unexpected}")
    if tau is None or c_phys is None or n_shots is None or backend is None:
        raise TypeError("tau, c_phys, n_shots, and backend are required")
    if used_legacy_names:
        warnings.warn(
            "Deprecated t is interpreted as tau (the evolution coefficient is "
            "2*pi*tau/mu), and deprecated C is interpreted as C_phys.",
            DeprecationWarning,
            stacklevel=2,
        )
    circuit = HHL_circuit(
        n_ancillas,
        b_vector,
        A_matrix,
        tau,
        c_phys,
        measure=False,
    )
    transpiled = transpile(circuit, backend, seed_transpiler=seed_transpiler)
    measured = add_solution_measurements(
        transpiled,
        n_ancillas,
        int(np.ceil(np.log2(len(b_vector)))),
    )
    counts = backend.run(
        measured,
        shots=n_shots,
        seed_simulator=seed_simulator,
    ).result().get_counts()
    n_state_qubits = int(np.ceil(np.log2(len(b_vector))))
    probabilities, _, _ = sampled_solution_probabilities(counts, n_state_qubits)
    return torch.from_numpy(probabilities)
