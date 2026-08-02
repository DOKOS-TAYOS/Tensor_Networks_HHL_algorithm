# README: Tensor Network HHL Simulation

## Project Title

**Solving Systems of Linear Equations: HHL from a Tensor Networks Perspective**

This repository accompanies the work presented in the paper:

> **"Solving Systems of Linear Equations: HHL from a Tensor Networks Perspective"**
> Alejandro Mata Ali and Iñigo Perez Delgado and Marina Ristol Roura and Aitor Moreno Fdez. de Leceta and Sebastián V. Romero (2025)
> [arXiv:2309.05290](https://arxiv.org/abs/2309.05290)

It can be consulted online in the Streamlit webpage: [https://tensornetworks-hhl-algorithm.streamlit.app/](https://tensornetworks-hhl-algorithm.streamlit.app/)

The project implements a classical simulation of the quantum Harrow-Hassidim-Lloyd (HHL) algorithm using tensor networks and qudit formalism. The goal is to provide a quantum-inspired solver that models the ideal behavior of HHL efficiently on classical hardware, enabling benchmarking and theoretical lower-bound estimations.

---

## Files

* `tensor_network_HHL.ipynb`: Jupyter notebook containing all code, explanations, experiments, and plots. It reproduces the results presented in the paper.

---

## Requirements

Install the dependencies with:

```bash
pip install numpy matplotlib scipy torch qiskit qiskit_ibm_runtime qiskit_aer
```

The notebook is compatible with standard Python 3.x and requires no GPU. All computations were tested on CPU.

---

## Usage

Open the notebook in Jupyter:

```bash
jupyter notebook tensor_network_HHL.ipynb
```

You may execute all cells sequentially to:

1. Define the tensor operations for the TN-HHL algorithm.
2. Construct tensors for QPE, inversion, and evolution operators.
3. Apply the method to benchmark problems:

   * Forced harmonic oscillator
   * Forced damped oscillator
   * 2D static heat equation with sources
4. Compare TN-HHL performance to:

   * Exact inversion (PyTorch)
   * Qiskit HHL simulation (for small cases)

Each section is self-contained and annotated for clarity.

---

## Summary of the Algorithm

* The notebook encodes the HHL quantum circuit using tensor networks.
* It implements all gates (QPE, inversion, unitaries) as tensor contractions.
* The final solution vector $\vec{x}$ is obtained deterministically, bypassing quantum limitations like post-selection.

### Parameter conventions

* `tau` (API name `t`) sets the spectral grid spacing: $\Delta\lambda = 1/\tau$.
* `mu` (API names `n_eigen` / `num_eigen`) is the phase-register dimension. It is **not** by itself the spectral resolution.
* The signed non-aliased spectral range depends on both parameters: $|\lambda| < \mu/(2\tau)$, equivalently $|\tau\lambda| < \mu/2$.
* `n_c` is used only for a binary phase register with $\mu = 2^{n_c}$. The general tensor implementation does not require $\mu$ to be a power of two.
* In the quantum-circuit HHL, `C` is the ancilla controlled-rotation scale constant and must satisfy $|C/\lambda_j| \le 1$ for all relevant eigenvalues.

### Fourier and inversion conventions

* The paper derivation uses the unnormalized Fourier matrix $H[a,b]=\exp(2\pi i\,ab/\mu)$, with $H^{-1}=H^\dagger/\mu$.
* The code uses the **normalized** QFT $F=H/\sqrt{\mu}$, which is unitary.
* The inverter tensor stores the global factor $\tau/\mu$ so that the normalized implementation remains equivalent to the paper's unnormalized formulation.

### Real-valued experimental scope

The numerical experiments accompanying the current paper use real-valued matrices and right-hand sides. The present public implementation returns real-valued solution vectors and has been validated only for this real-valued benchmark setting. The underlying tensor-network formulation is **not** restricted to real systems. Supporting genuinely complex-valued inputs would require retaining the complete complex output and using phase-aware validation metrics.

---

## Reproducibility

The notebook is reproducible and self-contained. It includes exact matrix definitions, right-hand sides, and benchmark hyperparameters.

---

## Reference

If you use this code, please cite the original paper (also encoded in [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{ali2024solvingsystemslinearequations,
      title={Solving Systems of Linear Equations: HHL from a Tensor Networks Perspective}, 
      author={Alejandro Mata Ali and Iñigo Perez Delgado and Marina Ristol Roura and Aitor Moreno Fdez. de Leceta and Sebastián V. Romero},
      year={2024},
      eprint={2309.05290},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2309.05290}, 
}
```

---

## License and copyright

This **software** is released under the MIT License. Copyright (c) 2025 Alejandro Mata Ali — see [`LICENSE`](LICENSE).

The accompanying **paper** ([arXiv:2309.05290](https://arxiv.org/abs/2309.05290)) lists additional coauthors (Iñigo Perez Delgado, Marina Ristol Roura, Aitor Moreno Fdez. de Leceta, and Sebastián V. Romero). Paper authorship is for scientific credit and citation; it does **not** change the software copyright holder named in `LICENSE`.

Third-party runtime dependencies (including Qiskit and PyTorch) are described in [`NOTICE`](NOTICE).