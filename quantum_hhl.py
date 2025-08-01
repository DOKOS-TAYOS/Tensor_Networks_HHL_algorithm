import torch

import numpy as np
from  scipy.linalg import expm


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import phase_estimation, UnitaryGate, RYGate


def HHL_circuit(n_ancillas: int, b_vector: torch.Tensor, A_matrix: torch.Tensor, t: float, C: float):
    """
    Constructs an HHL (Harrow-Hassidim-Lloyd) quantum circuit for solving linear systems Ax = b.
    
    This function implements the quantum algorithm for solving systems of linear equations
    using quantum phase estimation and controlled rotations. The circuit consists of:
    1. State preparation with input vector b
    2. Quantum phase estimation to extract eigenvalues
    3. Controlled rotations based on eigenvalue estimates
    4. Inverse phase estimation
    
    Args:
        n_ancillas (int): Number of ancilla qubits for phase estimation (determines precision)
        b_vector (torch.Tensor): Input vector b of the linear system Ax = b
        A_matrix (torch.Tensor): Hermitian matrix A of the linear system (must be square)
        t (float): Time parameter for unitary evolution operator U = exp(iAt)
        C (float): Scaling parameter for rotation angles (typically C ≤ min eigenvalue)
    
    Returns:
        QuantumCircuit: Complete HHL quantum circuit ready for execution
        
    Note:
        The circuit automatically pads inputs to power-of-2 dimensions for efficient
        quantum implementation. The solution vector x is encoded in the final state
        of the quantum register.
    """
    # Determine the number of qubits needed for the state
    n_state_qubits = int(np.ceil(np.log2(len(b_vector))))
    # Pad vector b to power of 2 dimensions
    b_padded = np.concatenate([b_vector.real.numpy(), np.zeros(2**n_state_qubits - len(b_vector))])
    
    # Pad matrix A to power of 2 dimensions with identity matrix
    A_padded = np.eye(2**n_state_qubits)
    A_padded[:A_matrix.shape[0], :A_matrix.shape[1]] = A_matrix.real.numpy()
    
    # Create unitary evolution operator
    U_matrix = expm(1j * A_padded * t)
    U_gate = UnitaryGate(U_matrix, label='U')
    
    # Define quantum registers
    ancilla_reg = QuantumRegister(1, 'Anc')
    clock_reg = QuantumRegister(n_ancillas, 'Clock')
    state_reg = QuantumRegister(n_state_qubits, 'State')

    # Define the classical registers
    crAncilla = ClassicalRegister(1, name='canc')
    crClock = ClassicalRegister(n_ancillas, name='cClock')
    crState = ClassicalRegister(n_state_qubits, name='cState')
    
    # Create quantum circuit
    q_circ = QuantumCircuit(ancilla_reg, clock_reg, state_reg, crAncilla, crClock, crState, name='HHL')
    
    # Initialize state register with normalized input vector
    q_circ.initialize(b_padded, state_reg, normalize=True)
    q_circ.barrier()
    
    # Apply Quantum Phase Estimation
    q_circ.append(phase_estimation(n_ancillas, U_gate), clock_reg[:] + state_reg[:])
    q_circ.barrier()

    # Apply controlled rotation gates for positive eigenvalues
    for lambda_i in range(1, 2**n_ancillas // 2):
        # Calculate rotation angle based on eigenvalue
        theta = 2 * np.arcsin(C / lambda_i)
        
        # Convert eigenvalue index to binary representation
        binary_str = bin(lambda_i)[2:]
        binary_str = ('0' * (n_ancillas - len(binary_str)) + binary_str)  # Pad with leading zeros
        
        # Apply X gates to create appropriate control pattern
        for j, bit in enumerate(binary_str):
            if bit == '0':
                q_circ.x(j + 1)  # +1 to account for ancilla register
        
        # Apply controlled rotation gate
        q_circ.append(RYGate(theta).control(n_ancillas), clock_reg[:] + ancilla_reg[:])
        
        # Undo X gates to restore original state
        for j, bit in enumerate(binary_str):
            if bit == '0':
                q_circ.x(j + 1)  # +1 to account for ancilla register

    for lambda_i in range(2**n_ancillas // 2, 2**n_ancillas):  # Negative eigenvalues
        # Calculate rotation angle based on eigenvalue
        theta = 2 * np.arcsin(-C / (2**n_ancillas - lambda_i))
        
        # Convert eigenvalue index to binary representation
        binary_str = bin(lambda_i)[2:]
        binary_str = ('0' * (n_ancillas - len(binary_str)) + binary_str)  # Pad with leading zeros
        
        # Apply X gates to create appropriate control pattern
        for j, bit in enumerate(binary_str):
            if bit == '0':
                q_circ.x(j + 1)  # +1 to account for ancilla register
        
        # Apply controlled rotation gate
        q_circ.append(RYGate(theta).control(n_ancillas), clock_reg[:] + ancilla_reg[:])
        
        # Undo X gates to restore original state
        for j, bit in enumerate(binary_str):
            if bit == '0':
                q_circ.x(j + 1)  # +1 to account for ancilla register

    q_circ.barrier()

    # Inverse QPE
    q_circ.append(phase_estimation(n_ancillas, U_gate).inverse(), clock_reg[:] + state_reg[:])

    q_circ.barrier()
    q_circ.measure(ancilla_reg, crAncilla)
    q_circ.measure(clock_reg, crClock)
    q_circ.measure(state_reg, crState)

    # We need to decompose QPE to avoid errors
    qc_desc = q_circ.decompose(['QPE', 'QPE_dg'], reps=2)

    # display(q_circ.draw('mpl'))

    return qc_desc

def original_HHL_solver(n_ancillas: int, b_vector: torch.Tensor, A_matrix: torch.Tensor, t: float, C: float,
                        n_shots: int, backend):
    """
    Execute the HHL algorithm on a simulator backend.
    
    Args:
        n_ancillas: Number of ancilla qubits for phase estimation
        b_vector: Input vector b in the equation Ax = b
        A_matrix: Hermitian matrix A in the equation Ax = b
        t: Time parameter for the evolution
        C: Normalization constant
        n_shots: Number of shots for the quantum circuit execution
        backend: Simulator backend to run the circuit on
        
    Returns:
        torch.Tensor: Vector of probabilities representing the solution
    """
    
    # Create the HHL quantum circuit
    qc_circ = HHL_circuit(n_ancillas, b_vector, A_matrix, t, C)

    # Transpile the circuit for the target backend
    qc_transpiled = transpile(qc_circ, backend)

    # Execute the circuit with specified number of shots
    job = backend.run(qc_transpiled, shots=n_shots)
    
    # Get measurement results
    counts = job.result().get_counts()
    
    # Calculate number of state qubits
    n_state = int(np.ceil(np.log2(len(b_vector))))

    # Extract successful measurements (ancilla qubit = '1')
    success_counts = {key[:-2]: counts[key] for key in counts if key[-1] == '1'}
    
    # Initialize probability dictionary for all possible states
    x_state = {}
    for i in range(2**n_state): # Start by initializing the keys
        binary = bin(i)[2:]
        binary = '0'*(n_state-len(binary)) + binary
        x_state[binary] = 0

    for i in range(2**n_state): # Sum all that match
        binary = bin(i)[2:]
        binary = '0'*(n_state-len(binary)) + binary
        x_state[binary] += sum([ success_counts[key] for key in success_counts if key[:n_state] == binary ])
    
    # Normalize probabilities
    total_sum = sum(x_state.values())
    if total_sum > 0:
        x_state = {key: value / total_sum for key, value in x_state.items()}

    # display(plot_histogram(x_state))

    # Convert x_state dictionary to a vector of probabilities
    probabilities = []
    for i in range(2**n_state):
        binary = bin(i)[2:]
        binary = '0'*(n_state-len(binary)) + binary
        probabilities.append(x_state[binary])
    probabilities = torch.tensor(probabilities, dtype=torch.float64)
    return probabilities



































