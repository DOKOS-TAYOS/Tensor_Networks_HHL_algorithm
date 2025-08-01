import torch
import numpy as np


def qft_op(n_eigen: int, sign: int) -> torch.Tensor:
    """
    Creates a Quantum Fourier Transform (QFT) matrix or its inverse.
    
    This function constructs the QFT matrix using the formula:
    QFT = (1/√m) * Σ_{x,y=0}^{m-1} exp(2πi * xy/m) |x⟩⟨y|
    
    For inverse QFT, the sign is negative: exp(-2πi * xy/m)
    
    Parameters:
    -----------
    n_eigen : int
        Size of the QFT matrix (n_eigen x n_eigen).
    sign : int
        Sign for the phase factor: +1 for QFT, -1 for inverse QFT.
    
    Returns:
    --------
    torch.Tensor
        The QFT matrix (normalized by 1/√n_eigen factor).
    
    Notes:
    ------
    - The matrix is constructed using a recurrence relation for efficiency
    - The result is normalized with the 1/√n_eigen prefactor
    """
    matrix = torch.ones((n_eigen, n_eigen), dtype=torch.complex128)
    # Phase factor for the QFT: 2πi/n_eigen (positive) or -2πi/n_eigen (negative)
    angle = 1j * sign * 2.0 * np.pi / n_eigen
    
    # Initialize the first row with phase factors (matrix[1,0] = 1 already set)
    for j in range(1, n_eigen):
        matrix[1, j] = np.exp(angle * j)
    
    # Use recurrence relation: row[i] = row[i-1] * row[1] (element-wise)
    # This exploits the periodicity of the QFT matrix structure
    for i in range(2, n_eigen):
        matrix[i] = matrix[i-1] * matrix[1]

    return matrix / np.sqrt(n_eigen)



def phase_kickback_op(b_vector:torch.Tensor, n_eigen:int, U_matrix: torch.Tensor) -> torch.Tensor:
    """
    Function that creates the Phase KickBack tensor for QPE.
    
    This function computes U^0*b, U^1*b, U^2*b, ..., U^(n_eigen-1)*b where b is the input vector.
    The result is stored in a matrix where each row represents a different power of U applied to b.
    
    Parameters:
    -----------
    b_vector : torch.Tensor
        Input vector to be transformed.
    n_eigen : int
        Number of eigenvalues (determines the number of powers to compute).
    U_matrix : torch.Tensor
        Unitary matrix to be raised to powers.
    
    Returns:
    --------
    torch.Tensor
        Matrix where each row i contains U^i * b_vector.
    """
    # Multiply the previous by U
    n_elems = U_matrix.shape[0]
    phase_kick_matrix = torch.zeros((n_eigen, n_elems), dtype=torch.complex128)
    phase_kick_matrix[0] = b_vector.clone()
    ans = b_vector.clone()  # Clone to avoid modifying the original

    for i in range(1, n_eigen):
        ans = torch.matmul(U_matrix, ans)
        phase_kick_matrix[i] = ans
    return phase_kick_matrix

def phase_kickback_op_inv(n_eigen:int, U_matrix: torch.Tensor) -> torch.Tensor:
    """
    Function that creates the inverse Phase KickBack tensor for QPE.
    
    This function computes U^0, U^1, U^2, ..., U^(n_eigen-1) where U is the unitary matrix.
    The result is stored in a 3D tensor where each slice represents a different power of U.
    
    Parameters:
    -----------
    n_eigen : int
        Number of eigenvalues (determines the number of powers to compute).
    U_matrix : torch.Tensor
        Unitary matrix to be raised to powers.
    
    Returns:
    --------
    torch.Tensor
        3D tensor where each slice i contains U^i.
    """
    # Multiply the previous by U
    n_elems = U_matrix.shape[0]
    phase_kick_tensor = torch.zeros((n_eigen, n_elems, n_elems), dtype=torch.complex128)
    phase_kick_tensor[0] = torch.eye(n_elems, dtype=torch.complex128)
    ans = phase_kick_tensor[0].clone()  # Clone to avoid modifying the original

    for i in range(1, n_eigen):
        ans = torch.matmul(ans, U_matrix)
        phase_kick_tensor[i] = ans
    return phase_kick_tensor

def inversor(n_eigen: int, t: float) -> torch.Tensor:
    """
    Function that creates the inversion matrix.
    
    This function creates a diagonal matrix with eigenvalues 1/i and -1/i,
    where i ranges from 1 to n_eigen//2. This is used for eigenvalue inversion
    in the HHL algorithm.
    
    Parameters:
    -----------
    n_eigen : int
        Size of the square matrix to create.
    t : float
        Exponential scaling factor.
    
    Returns:
    --------
    torch.Tensor
        The matrix with inversion values on the diagonal.
    """
    matrix = torch.zeros((n_eigen, n_eigen), dtype=torch.complex128)
    # Assign the values
    for i in range(1, (n_eigen // 2 + 1)):  # The +1 is for the extra in case n_eigen is even.
        matrix[-i, -i] = -1 / i  # Negative eigenvalues
        matrix[i, i] = 1 / i     # Positive eigenvalues

    return matrix * t / n_eigen



def tracer(W_matrix: torch.Tensor, U_matrix: torch.Tensor):
    '''
    This function performs the trace after the last product.
    Parameters:
    - W_matrix, torch.Tensor: W matrix.
    - U_matrix, torch.Tensor: inverse U tensor.
    Returns:
    - C_matrix, torch.Tensor: tensor with unscaled solution.
    '''
    C_matrix = torch.tensordot(W_matrix, U_matrix, dims=[[0,1], [0,1]])  # C_k = W_ij U^(-1)_ijk

    return C_matrix
#-----------------------------------------------------------------
def tensornetwork_HHL(num_eigen: int, t: float, b_vector: torch.Tensor, A_matrix: torch.Tensor):
    '''
    Function that generates tensors and contracts them.
    Parameters:
    - num_eigen, int: number of eigenvalues.
    - t, float: exponential scaling factor.
    - b_vector, torch.Tensor: vector b.
    - A_matrix, torch.Tensor: matrix A.
    Returns:
    - solution, torch.Tensor: the solution x
    '''
    # Calculate dimensions
    n_elements   = len(b_vector)
    # Calculate complex exponentials
    U_matrix_inv = torch.matrix_exp(-(2j*np.pi*t/num_eigen)*A_matrix)
    U_matrix     = torch.conj(U_matrix_inv).T  # its transpose

    # Product PKB-QFT_inverse-inversor-QFT, careful with
    # indices, done to avoid transpositions
    W_matrix = torch.matmul( qft_op(num_eigen, sign=1),     \
                   torch.matmul( inversor(num_eigen, t), \
                        torch.matmul(qft_op(num_eigen, sign=-1), \
                            phase_kickback_op(b_vector, num_eigen, U_matrix)) ) )
    # Rescaled result
    return tracer(W_matrix, phase_kickback_op_inv(num_eigen, U_matrix_inv)).real






























































