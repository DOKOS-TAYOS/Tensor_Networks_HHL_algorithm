"""Mathematical convention tests for TN-HHL (referee comments 1 and 3)."""
from __future__ import annotations

import unittest

import torch

from tn_hhl import (
    _require_effectively_real,
    inversor,
    qft_op,
    signed_phase_index,
    tensornetwork_HHL,
)


class TestQFTConventions(unittest.TestCase):
    def test_normalized_qft_unitarity(self) -> None:
        for mu in (3, 4, 5, 8):
            with self.subTest(mu=mu):
                F = qft_op(mu, sign=1)
                identity = torch.eye(mu, dtype=torch.complex128)
                self.assertTrue(
                    torch.allclose(F.conj().T @ F, identity, atol=1e-12, rtol=1e-12)
                )

    def test_negative_phase_is_adjoint(self) -> None:
        for mu in (3, 4, 7):
            with self.subTest(mu=mu):
                F_positive = qft_op(mu, sign=1)
                F_negative = qft_op(mu, sign=-1)
                self.assertTrue(
                    torch.allclose(
                        F_negative,
                        F_positive.conj().T,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                )

    def test_unnormalized_adjoint_vs_inverse(self) -> None:
        for mu in (3, 4, 6):
            with self.subTest(mu=mu):
                F = qft_op(mu, sign=1)
                H = torch.sqrt(torch.tensor(float(mu), dtype=torch.float64)) * F
                self.assertTrue(
                    torch.allclose(
                        torch.linalg.inv(H),
                        H.conj().T / mu,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                )


class TestInversorScaling(unittest.TestCase):
    def test_normalized_unnormalized_equivalence(self) -> None:
        for mu, tau in ((4, 2.5), (5, 1.0), (7, 3.0)):
            with self.subTest(mu=mu, tau=tau):
                F = qft_op(mu, sign=1)
                H = torch.sqrt(torch.tensor(float(mu), dtype=torch.float64)) * F

                G = torch.zeros((mu, mu), dtype=torch.complex128)
                for d in range(mu):
                    signed_d = signed_phase_index(d, mu)
                    if signed_d != 0:
                        G[d, d] = 1.0 / signed_d

                scaled = inversor(mu, tau)
                self.assertTrue(
                    torch.allclose(scaled, (tau / mu) * G, atol=1e-12, rtol=1e-12)
                )

                # Contraction order in tensornetwork_HHL:
                # F @ ((tau/mu)*G) @ F.conj().T
                normalized = F @ ((tau / mu) * G) @ F.conj().T
                unnormalized = (tau / mu**2) * H @ G @ H.conj().T
                self.assertTrue(
                    torch.allclose(normalized, unnormalized, atol=1e-12, rtol=1e-12)
                )


class TestSignedPhaseIndex(unittest.TestCase):
    def test_signed_bins(self) -> None:
        mu = 8
        self.assertEqual(signed_phase_index(0, mu), 0)
        self.assertEqual(signed_phase_index(1, mu), 1)
        self.assertEqual(signed_phase_index(3, mu), 3)
        self.assertEqual(signed_phase_index(7, mu), -1)
        self.assertEqual(signed_phase_index(6, mu), -2)
        # Nyquist for even mu treated as positive
        self.assertEqual(signed_phase_index(mu // 2, mu), mu // 2)

    def test_odd_mu(self) -> None:
        mu = 5
        self.assertEqual(signed_phase_index(0, mu), 0)
        self.assertEqual(signed_phase_index(2, mu), 2)
        self.assertEqual(signed_phase_index(3, mu), -2)
        self.assertEqual(signed_phase_index(4, mu), -1)


class TestRealInputValidation(unittest.TestCase):
    def test_require_effectively_real_accepts_real(self) -> None:
        _require_effectively_real(torch.tensor([1.0, 2.0]), name="x")

    def test_require_effectively_real_accepts_zero_imag(self) -> None:
        _require_effectively_real(
            torch.tensor([1.0 + 0.0j, 2.0 + 0.0j]), name="x"
        )

    def test_require_effectively_real_rejects_complex(self) -> None:
        with self.assertRaises(ValueError):
            _require_effectively_real(
                torch.tensor([1.0 + 1e-6j]), name="x", atol=1e-12
            )

    def test_tensornetwork_hhl_rejects_complex_A(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.complex128)
        A = A + 1e-3j * torch.eye(2, dtype=torch.complex128)
        b = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        with self.assertRaises(ValueError):
            tensornetwork_HHL(8, 1.0, b, A)

    def test_tensornetwork_hhl_rejects_complex_b(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.complex128)
        b = torch.tensor([1.0 + 1e-3j, 0.0], dtype=torch.complex128)
        with self.assertRaises(ValueError):
            tensornetwork_HHL(8, 1.0, b, A)

    def test_tensornetwork_hhl_accepts_real_and_zero_imag(self) -> None:
        A_real = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
        b_real = torch.tensor([1.0, -0.5], dtype=torch.float64)
        x_real = tensornetwork_HHL(32, 8.0, b_real, A_real)

        A_c = A_real.to(torch.complex128)
        b_c = b_real.to(torch.complex128)
        x_c = tensornetwork_HHL(32, 8.0, b_c, A_c)
        self.assertTrue(torch.allclose(x_real, x_c, atol=1e-10, rtol=1e-10))


class TestIntegralSmallSystem(unittest.TestCase):
    def test_small_symmetric_real_system(self) -> None:
        A = torch.tensor(
            [[3.0, 0.5, 0.0], [0.5, 2.0, 0.25], [0.0, 0.25, 1.5]],
            dtype=torch.float64,
        )
        b = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64)
        x = tensornetwork_HHL(200, 20.0, b, A)

        self.assertEqual(tuple(x.shape), (3,))
        self.assertFalse(torch.is_complex(x))
        self.assertFalse(torch.any(torch.isnan(x)))
        self.assertFalse(torch.any(torch.isinf(x)))

        x_exact = torch.linalg.solve(A, b)
        # Preserve prior real-valued behaviour within numerical tolerance
        self.assertTrue(torch.allclose(x, x_exact, atol=5e-3, rtol=5e-3))
        residual = A @ x - b
        self.assertLess(torch.linalg.norm(residual).item(), 5e-2)


if __name__ == "__main__":
    unittest.main()
