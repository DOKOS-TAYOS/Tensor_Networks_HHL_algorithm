# README: Tensor Network HHL Simulation

## Project Title

**Solving Systems of Linear Equations: HHL from a Tensor Networks Perspective**

This repository accompanies the work presented in the paper:

> **"Solving Systems of Linear Equations: HHL from a Tensor Networks Perspective"**
> Alejandro Mata Ali and Iñigo Perez Delgado and Marina Ristol Roura and Aitor Moreno Fdez. de Leceta and Sebastián V. Romero (2025)
> [arXiv:2309.05290](https://arxiv.org/abs/2309.05290)

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
* Eigenvalue resolution (parameter `mu`) and time evolution (parameter `tau`) are tunable.
* The final solution vector \$\vec{x}\$ is obtained deterministically, bypassing quantum limitations like post-selection.

---

## Reproducibility

The notebook is reproducible and self-contained. It includes exact matrix definitions, right-hand sides, and benchmark hyperparameters.

---

## Reference

If you use this code, please cite the original paper:

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

## License

MIT License (c) Alejandro Mata Ali, 2025
