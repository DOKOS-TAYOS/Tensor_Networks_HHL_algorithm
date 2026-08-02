import torch
import numpy as np


def signed_phase_index(d: int, mu: int) -> int:
    """Map a phase-register bin index to its signed spectral representative.

    Convention (paper):
        signed_d = d if d <= mu // 2 else d - mu

    - Bin ``d = 0`` is the singular bin (signed value 0).
    - For even ``mu``, the Nyquist index ``d = mu // 2`` is treated as positive.
    """
    if d <= mu // 2:
        return d
    return d - mu


def _require_effectively_real(
    tensor: torch.Tensor,
    *,
    name: str,
    atol: float = 1e-12,
) -> None:
    """Reject genuinely complex inputs in the current real-valued implementation."""
    if not torch.is_complex(tensor):
        return
    if torch.max(torch.abs(tensor.imag)).item() <= atol:
        return
    raise ValueError(
        f"{name} has a non-negligible imaginary part. "
        "The present public implementation is validated only for the "
        "real-valued numerical experiments accompanying the paper."
    )


def qft_op(n_eigen: int, sign: int) -> torch.Tensor:
    """
    Build a normalized Quantum Fourier Transform (QFT) matrix.

    Historical API note: ``n_eigen`` represents the phase-register dimension
    ``mu`` (not a count of eigenvalues).

    The analytic derivation in the paper uses the unnormalized Fourier matrix
        H[a, b] = exp(2j * pi * a * b / mu),
    which satisfies ``H.conj().T @ H == mu * I`` and therefore
        H^{-1} = H^\\dagger / mu
    (the adjoint alone is not the inverse).

    This implementation returns the unitary (normalized) QFT
        F = H / sqrt(mu),
    which satisfies ``F.conj().T @ F == I``.

    Parameters
    ----------
    n_eigen : int
        Phase-register dimension ``mu``. Size of the returned ``mu x mu`` matrix.
    sign : int
        Phase orientation: ``+1`` yields ``F`` with entries
        ``exp(+2j*pi*a*b/mu) / sqrt(mu)``; ``-1`` yields the negative-phase
        orientation ``exp(-2j*pi*a*b/mu) / sqrt(mu)``, which is exactly
        ``F.conj().T`` (the adjoint of the positive orientation). Do not call
        the negative-phase matrix an ``H``-inverse: the true inverse of the
        unnormalized ``H`` is ``H^\\dagger / mu``.

    Returns
    -------
    torch.Tensor
        Normalized Fourier matrix ``F`` with dtype ``torch.complex128``.

    Notes
    -----
    - The matrix is constructed using a recurrence relation for efficiency.
    - Operations remain in complex precision even when inputs ``A`` and ``b``
      are real-valued.
    """
    mu = n_eigen
    matrix = torch.ones((mu, mu), dtype=torch.complex128)
    # Phase factor: +2πi/μ (sign=+1) or -2πi/μ (sign=-1)
    angle = 1j * sign * 2.0 * np.pi / mu

    for j in range(1, mu):
        matrix[1, j] = np.exp(angle * j)

    # Recurrence: row[i] = row[i-1] * row[1] (element-wise)
    for i in range(2, mu):
        matrix[i] = matrix[i - 1] * matrix[1]

    return matrix / np.sqrt(mu)


def phase_kickback_op(
    b_vector: torch.Tensor, n_eigen: int, U_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Create the Phase KickBack tensor for QPE.

    Computes ``U^0 b, U^1 b, ..., U^(mu-1) b`` where ``b`` is the input vector.
    Each row of the result stores one power applied to ``b``.

    Parameters
    ----------
    b_vector : torch.Tensor
        Input vector to be transformed.
    n_eigen : int
        Phase-register dimension ``mu`` (historical name; not a count of
        eigenvalues). Determines how many powers of ``U`` are computed.
    U_matrix : torch.Tensor
        Unitary matrix raised to successive powers.

    Returns
    -------
    torch.Tensor
        Matrix where row ``i`` contains ``U^i * b_vector``.
    """
    mu = n_eigen
    n_elems = U_matrix.shape[0]
    phase_kick_matrix = torch.zeros((mu, n_elems), dtype=torch.complex128)
    phase_kick_matrix[0] = b_vector.clone()
    ans = b_vector.clone()

    for i in range(1, mu):
        ans = torch.matmul(U_matrix, ans)
        phase_kick_matrix[i] = ans
    return phase_kick_matrix


def phase_kickback_op_inv(n_eigen: int, U_matrix: torch.Tensor) -> torch.Tensor:
    """
    Create the inverse Phase KickBack tensor for QPE.

    Computes ``U^0, U^1, ..., U^(mu-1)`` where ``U`` is the unitary matrix.
    Each slice of the result stores one power of ``U``.

    Parameters
    ----------
    n_eigen : int
        Phase-register dimension ``mu`` (historical name; not a count of
        eigenvalues). Determines how many powers of ``U`` are computed.
    U_matrix : torch.Tensor
        Unitary matrix raised to successive powers.

    Returns
    -------
    torch.Tensor
        3D tensor where slice ``i`` contains ``U^i``.
    """
    mu = n_eigen
    n_elems = U_matrix.shape[0]
    phase_kick_tensor = torch.zeros((mu, n_elems, n_elems), dtype=torch.complex128)
    phase_kick_tensor[0] = torch.eye(n_elems, dtype=torch.complex128)
    ans = phase_kick_tensor[0].clone()

    for i in range(1, mu):
        ans = torch.matmul(ans, U_matrix)
        phase_kick_tensor[i] = ans
    return phase_kick_tensor


def inversor(n_eigen: int, t: float) -> torch.Tensor:
    """
    Build the scaled diagonal inversion matrix used with normalized QFTs.

    Historical API notes:
    - ``n_eigen`` represents the phase-register dimension ``mu``.
    - ``t`` represents the spectral-resolution parameter ``tau``.

    Let ``G`` be the unscaled diagonal with
        G[d, d] = 1 / signed_d    for signed_d != 0,
        G[0, 0] = 0,
    using ``signed_phase_index``. For even ``mu``, the Nyquist bin
    ``d = mu // 2`` is treated as positive.

    This function returns ``(tau / mu) * G``. Combined with the normalized
    Fourier matrices ``F = H / sqrt(mu)`` in the contraction order of
    ``tensornetwork_HHL``,
        F @ ((tau / mu) * G) @ F.conj().T
    equals
        (tau / mu**2) * H @ G @ H.conj().T,
    which matches the unnormalized paper formulation.

    Parameters
    ----------
    n_eigen : int
        Phase-register dimension ``mu``.
    t : float
        Spectral-resolution parameter ``tau`` (grid spacing ``Delta lambda = 1/tau``).

    Returns
    -------
    torch.Tensor
        Diagonal complex matrix ``(tau / mu) * G``.
    """
    mu = n_eigen
    tau = t
    matrix = torch.zeros((mu, mu), dtype=torch.complex128)
    for d in range(mu):
        signed_d = signed_phase_index(d, mu)
        if signed_d != 0:
            matrix[d, d] = 1.0 / signed_d

    return matrix * tau / mu


def tracer(W_matrix: torch.Tensor, U_matrix: torch.Tensor) -> torch.Tensor:
    """
    Contract the final product and take the partial trace over the phase register.

    Parameters
    ----------
    W_matrix : torch.Tensor
        Intermediate W matrix after QFT / inversion / phase-kickback contraction.
    U_matrix : torch.Tensor
        Inverse phase-kickback tensor.

    Returns
    -------
    torch.Tensor
        Unscaled complex solution tensor before taking the real part.
    """
    # C_k = W_ij U^(-1)_ijk
    return torch.tensordot(W_matrix, U_matrix, dims=[[0, 1], [0, 1]])


def tensornetwork_HHL(
    num_eigen: int, t: float, b_vector: torch.Tensor, A_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Run the tensor-network HHL contraction and return a real-valued solution.

    Historical API notes:
    - ``num_eigen`` / ``n_eigen`` represents the phase-register dimension ``mu``.
    - ``t`` represents the spectral-resolution parameter ``tau``.
      Spectral spacing is ``Delta lambda = 1 / tau``; the signed non-aliased
      range satisfies ``|lambda| < mu / (2 * tau)``.
    - ``n_c`` appears only for binary phase registers with ``mu = 2**n_c``; this
      general tensor implementation does not require ``mu`` to be a power of two.

    The numerical experiments accompanying the current paper use real-valued
    matrices and right-hand sides. The present public implementation returns
    real-valued solution vectors and has been validated only for this
    real-valued benchmark setting. The underlying tensor-network formulation
    is not restricted to real systems. Supporting genuinely complex-valued
    inputs would require retaining the complete complex output and using
    phase-aware validation metrics.

    Parameters
    ----------
    num_eigen : int
        Phase-register dimension ``mu``.
    t : float
        Spectral-resolution parameter ``tau``.
    b_vector : torch.Tensor
        Right-hand side ``b`` (must be effectively real).
    A_matrix : torch.Tensor
        Coefficient matrix ``A`` (must be effectively real).

    Returns
    -------
    torch.Tensor
        Real-valued approximate solution ``x``.
    """
    _require_effectively_real(A_matrix, name="A_matrix")
    _require_effectively_real(b_vector, name="b_vector")

    mu = num_eigen
    tau = t
    # Keep internal evolution / QFT / contractions in complex128 even when
    # the validated inputs were stored as real tensors.
    A_c = A_matrix.to(dtype=torch.complex128)
    b_c = b_vector.to(dtype=torch.complex128)

    U_matrix_inv = torch.matrix_exp(-(2j * np.pi * tau / mu) * A_c)
    U_matrix = torch.conj(U_matrix_inv).T

    # Order: F @ ((tau/mu)*G) @ F^† @ phase_kickback  (indices chosen to avoid
    # extra transpositions). F = qft_op(..., sign=+1), F^† = qft_op(..., sign=-1).
    W_matrix = torch.matmul(
        qft_op(mu, sign=1),
        torch.matmul(
            inversor(mu, tau),
            torch.matmul(
                qft_op(mu, sign=-1),
                phase_kickback_op(b_c, mu, U_matrix),
            ),
        ),
    )
    # Keep the real part for the validated real-valued experimental setting.
    return tracer(W_matrix, phase_kickback_op_inv(mu, U_matrix_inv)).real
