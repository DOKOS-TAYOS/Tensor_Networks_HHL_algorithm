# Scientific additions for Referee 1 comments 5 and 6

This note collects scientific points that should accompany the new numerical tables and figures when the manuscript and response letter are revised. It complements the tables and figures in `artifacts/reviewer_r1_c5_c6/`; numerical values should be taken from the generated CSV files after the final experiment run.

## Scope of the TN HHL contribution

The defensible role of the tensor-network method is a classical, explicit simulator of the ideal HHL transformation for matrix sizes that remain classically tractable. It is not presented as competitive with optimized direct or iterative classical linear solvers. Its scientific use is to expose the HHL/QPE structure, separate spectral-discretization effects from hardware noise, and study how the output changes with the phase-register dimension, spectral scale, matrix spectrum, and condition number.

After extending the admissible phase-register dimensions, the three physical examples can be solved without aliasing and with a target relative filter error below 1%. Errors at this level are adequate for demonstrating that the TN contraction realizes the expected ideal HHL spectral map. This statement concerns numerical fidelity, not computational competitiveness.

The original notebook values described as RMSE were computed as an absolute residual divided by the system dimension,

\[
\frac{\lVert b-A\widehat{x}\rVert_2}{N},
\]

and, for the oscillator cells, included an additional `dt**2` scaling. They are not component-wise solution RMSE. The revised experiments report, without conflating them:

\[
\operatorname{RMSE}(\widehat{x},x)=
\frac{\lVert\widehat{x}-x\rVert_2}{\sqrt{N}},
\qquad
e_{\mathrm{rel}}=
\frac{\lVert\widehat{x}-x\rVert_2}{\lVert x\rVert_2},
\qquad
r_{\mathrm{norm}}=
\frac{\lVert A\widehat{x}-b\rVert_2}{\lVert b\rVert_2}.
\]

Consequently, the new RMSE values must not be compared numerically with the former residual-based values as if they represented the same quantity.

## Exact spectral selector: purpose and limitation

The parameter-selection method is explicitly labeled `exact_spectral_benchmark_oracle`. It performs a full Hermitian eigendecomposition and evaluates the right-hand-side-specific error

\[
E_b=\frac{\lVert f_{\mu,\tau}(A)b-A^{-1}b\rVert_2}{\lVert A^{-1}b\rVert_2}
\]

over a predefined grid. This is appropriate as an offline calibration procedure for controlled, classically tractable HHL benchmarks because it removes visual hand tuning, supplies ground truth, and makes the selected pair reproducible.

It is not a realistic deployable selector for an unknown large problem. Computing `E_b` requires enough spectral information to construct the classical reference solution. If the sole objective were to obtain `x`, a direct classical solve would make the subsequent TN execution unnecessary. A practical parameter rule would instead require prior physical knowledge, certified spectral bounds, or iterative estimators such as Lanczos, and it could not use the unknown `A^{-1}b` as its criterion.

The calibration cost is recorded separately from TN and reference-solver timings. For `G` evaluated candidates, the present implementation has the approximate cost

\[
C_{\mathrm{selection}}=
O(N^3)+
O\left(\sum_{c=1}^{G}(N\mu_c+N^2)\right),
\]

where the first term is the eigendecomposition. This term should not be silently folded into the stated TN contraction complexity

\[
C_{\mathrm{TN}}=O(N^2\mu)+O(N\mu^2)+O(\mu^3).
\]

If end-to-end automatic solution is discussed, both costs must be stated. If the discussion concerns contraction complexity for fixed `(mu, tau)`, the calibration must be identified as separate benchmark preprocessing.

## Why the spectral filter does not replace the TN study

Once the full eigendecomposition is available, applying

\[
\widehat{x}=Vf_{\mu,\tau}(\Lambda)V^\dagger b
\]

is a simpler classical way to obtain the same ideal output. Therefore, the filter evaluation should not be presented as evidence that the TN contraction itself is computationally useful for solving `Ax=b`.

The filter and TN serve different research functions:

- the spectral filter characterizes the scalar transfer function applied by ideal HHL to each eigenvalue;
- the TN contraction explicitly represents QFT, inversion, phase kickback, contraction order, intermediate tensors, memory use, and possible future truncations or modifications;
- agreement between both is a validation of the TN representation, not a competition between two classical solvers.

If the paper claimed only that TN is another efficient classical method for returning `A^{-1}b`, the availability of the spectral evaluation would substantially weaken the contribution. The more precise claim is that TN is an explicit research simulator of ideal HHL and its internal structure.

## Sensitivity analysis and full-TN validation

The complete sensitivity grid contains 81 `(tau, n_c)` points evaluated on 20 random matrices, or 1,620 outputs. It includes `mu=2**13=8192`. Executing the current dense TN contraction at every point is not practical because it materializes dense phase-register objects and includes an `O(mu**3)` term.

The complete grid is therefore an ideal spectral-filter sensitivity analysis, not a full-TN runtime sweep. Every row is labeled `spectral_filter_equivalent`. It reports uncertainty across the 20 matrices and reveals phase-grid, aliasing, and condition-number effects over the complete predefined grid.

To support the use of this surrogate, a versioned representative subset is also run with the complete TN contraction. The subset spans multiple phase-register sizes, spectral scales, condition regimes, and includes a deliberately aliased point. `tn_filter_validation.csv` reports the TN/filter discrepancy, accuracy and residual of both outputs, aliasing status, and separate execution times. The full sensitivity figures may be used as properties of the ideal HHL filter only after this distinction is stated explicitly.

The main experimental evidence is not limited to the surrogate sweep: the three physical applications and all 20 selected random instances are executed with the complete TN contraction.

### Interpretation of the two sensitivity figures

The two figures contain the same 81 mean-RMSE values but expose complementary effects. In the plot against `tau`, a fixed phase-register size improves only while `tau` increases the useful spectral resolution without causing aliasing. The best non-aliased grid value consequently moves with register size: `tau=10` for `n_c=5,6`, `tau=50` for `n_c=7`, `tau=100` for `n_c=8`, `tau=200` for `n_c=9`, `tau=500` for `n_c=10`, and `tau=1000` for `n_c=11,12,13`. At `tau=10000`, every tested register size aliases at least one eigenvalue and the mean RMSE rises rather than improves.

The plot against `n_c` shows the converse statement. Increasing `n_c` has almost no effect when `tau` is already too small: for `tau<=10`, the curves are essentially flat from `n_c=5` to `13`. For larger `tau`, increasing `n_c` first removes aliasing and resolves the spectrum, after which the error reaches a plateau. For example, the sharp improvement for `tau=500` occurs at `n_c=10`, whereas for `tau=1000` it occurs at `n_c=11`. Increasing `n_c` beyond the plateau adds phase-register cost without a material accuracy gain at fixed `tau`.

The smallest mean RMSE on the predefined grid is `0.0418246103` at `(n_c,tau)=(13,1000)`, but the result has already saturated: the values at `n_c=11` and `12` are `0.0418255485` and `0.0418246269`. The relevant conclusion is therefore not that the largest register is intrinsically optimal, but that `tau` and `n_c` must be increased together until the useful non-aliased resolution has converged.

The across-instance distribution is strongly right-skewed. Fifty-six of the 81 two-sided Student-t intervals for the mean have a nonpositive lower bound, even though RMSE itself is nonnegative. This is caused by a few difficult matrices rather than by numerical zeros. For example, at `(n_c,tau)=(9,100)`, the median RMSE is `0.0198`, but instance 14 gives approximately `18.30`; this instance has the largest condition number in the set, approximately `469.8`. The resulting mean and sample deviation are approximately `0.962` and `4.08`.

On a logarithmic axis, a nonpositive confidence limit cannot be drawn. The `RMSE`-versus-`tau` figure therefore omits only the lower arm for such intervals, retains the upper arm, and marks aliased points with crosses. This is a display convention, not a conversion to a one-sided interval. The exact signed Student-t limits remain in `hyperparameter_sweep.csv`, and the omitted-arm count is stated in the figure. The large uncertainty is itself a scientific result: sensitivity depends not only on `(tau,n_c)` but also strongly on the spectrum and condition number of the instance.

## Extended damped-oscillator grid

The former maximum `mu=2000` could not meet the 1% target for the damped oscillator while preserving no aliasing. The approved grid includes `mu=4096`; the same predefined selection rule can then select a no-aliasing candidate below the target. This is a declared, versioned extension rather than an a posteriori manual choice.

The increased accuracy has a substantial resource cost. The selected damped run should be discussed as evidence of the accuracy/resource trade-off: increasing the phase-register dimension improves the spectral approximation but increases the contraction proxy and dense-memory requirements sharply.

## Numerical evidence from the extended run

The complete extended run finishes successfully with no parameter-selection warning. For the damped oscillator, the predefined rule selects

\[
\mu=4096,\qquad \tau=8892.098602882095,
\]

and the complete TN contraction gives

\[
\operatorname{RMSE}=0.1251940076,
\qquad
e_{\mathrm{rel}}=0.0065547973,
\qquad
r_{\mathrm{norm}}=0.0019019843.
\]

Thus the actual TN execution, not only the filter prediction, satisfies the 1% relative-error target without aliasing.

The representative validation comprises 18 complete TN/filter comparisons: three random matrices at six `(n_c, tau)` points. Fifteen comparisons are non-aliased and three deliberately exercise an aliased point. The maximum observed relative discrepancy between the complete TN output and the spectral-filter output is approximately

\[
2.17\times10^{-12}.
\]

This validates use of the spectral filter as a numerical surrogate for the ideal TN output while preserving the distinction between a spectral evaluation and an executed tensor contraction.

The measured timings also show why calibration must be reported separately. In the verified run, selector time versus median TN time is approximately:

| Problem | Selector | TN contraction |
|---|---:|---:|
| Harmonic oscillator | 1.0839 s | 0.3479 s |
| Damped oscillator | 1.8574 s | 2.0802 s |
| Heat equation | 4.7182 s | 2.1554 s |

For most small random instances the exact selector is also slower than the selected TN contraction; for the largest selected `mu`, TN becomes comparable or dominant. These measurements support two simultaneous statements: the exact selector is acceptable as offline research calibration, but it is not a credible low-overhead component of a practical end-to-end solver. Timings are machine-dependent and should be taken from the regenerated artifacts when final tables are prepared.

## Notebook TN-Qiskit comparison

The corrected notebook comparison uses the same 20 deterministic dimension-16 random systems and seed `12345` as the other reviewer experiments. For each instance, the selector is rerun on the predefined binary-register candidates

\[
\mu\in\{128,256,512,1024\}.
\]

TN and Qiskit then use exactly the same selected `(mu, tau)` and

\[
U=\exp\left(\frac{2\pi i\tau}{\mu}A\right).
\]

The physical rotation constant and its finite-bin counterpart are

\[
C_{\mathrm{phys}}=0.9\min_j|\lambda_j|,
\qquad
C_{\mathrm{bin}}=\tau C_{\mathrm{phys}}.
\]

All selected pairs are non-aliased, no relevant eigenvalue is assigned to the singular bin, and every nonzero-bin rotation satisfies `abs(C_bin/d_tilde) <= 1` without clipping. The stricter rotation-domain requirement means that only 7 of the 20 pairs meet the separate 1% spectral-filter target. This loss of direct-solve accuracy is reported rather than hidden; it does not invalidate the paired implementation comparison.

The primary Qiskit result uses `AerSimulator(method="statevector")` without measurements. After conditioning on the successful rotation ancilla, the phase register is traced out. Fidelity is therefore evaluated between the normalized TN state and the generally mixed reduced Qiskit solution state. The mean and median fidelities are `0.952200` and `0.964896`, respectively; the fidelity IQR is `[0.929390, 0.981097]`. The observed mean probability RMSE values are

\[
\operatorname{RMSE}(p_{\mathrm{TN}},p_{\mathrm{direct}})=1.515\times10^{-3},
\qquad
\operatorname{RMSE}(p_{\mathrm{Qiskit}},p_{\mathrm{direct}})=5.428\times10^{-3},
\]

with a mean exact TN-Qiskit probability RMSE of `5.328e-3`. The secondary seeded 100,000-shot comparison has a mean sampled-versus-exact RMSE of `3.448e-3`; shot execution is excluded from the primary exact total.

With one warm-up, three measured repetitions, their per-instance medians, complex128 arithmetic, and one CPU thread, the total exact speed ratio has median `72.269x`, IQR `36.278x` (`Q1=45.525x`, `Q3=81.804x`), mean `69.143x`, and sample deviation `30.139x`. TN is faster in all 20 instances.

The total difference is not caused by shots, which are excluded from the primary ratio. The median Qiskit circuit-construction-plus-transpilation time is `0.4700 s`, approximately 65% of its exact total, while median statevector simulation is `0.2440 s`, approximately 34%. Thus construction and transpilation are the largest measured source of the total difference.

No `speedup_core` is reported. TN's tensor-preparation phase performs repeated applications of `U` and `U^{-1}` that Aer performs during statevector execution, so Aer simulation divided only by TN's final contraction is not a like-for-like comparison. The CSV retains every phase timing, sets `speedup_core_valid=False`, and leaves `speedup_core` empty.

These results support a consistent implementation-level speedup for the tested finite HHL map. They do not establish quantum advantage and do not compare TN with optimized classical linear solvers. Both paths are classical: one directly contracts the qudit/TN representation and the other simulates a gate-level circuit with Qiskit Aer. The conclusion is limited to 20 dimension-16 real symmetric systems, `mu <= 1024`, the selected Aer and PyTorch versions, one machine, and the reported single-thread configuration. Residual phase-register entanglement also makes the exact Qiskit solution mixed, so fidelity need not be one even when both paths use identical spectral parameters. Complete per-instance values and phase timings are in `artifacts/reviewer_r1_c7_qiskit_comparison.csv`.

## Recommended manuscript framing

The numerical revision should make the following distinctions explicit:

1. TN HHL is evaluated as a simulator and diagnostic representation of ideal HHL, not as a faster replacement for optimized classical solvers.
2. The exact spectral selector is an offline benchmark oracle used to remove hand tuning; it is not an end-to-end practical selector.
3. Selector time, reference-solver time, spectral-filter time, and TN time are reported separately.
4. The 1,620-point study uses the algebraically equivalent ideal filter and is labeled accordingly.
5. A representative subset uses the complete TN contraction and validates the spectral surrogate numerically.
6. The physical examples and 20 selected random instances are complete TN executions.
7. The filter does not demonstrate TN speed; it supplies an independent reference for the transformation that TN is intended to realize.
8. Future work on practical parameter selection should replace exact diagonalization and `A^{-1}b` with spectral bounds or iterative estimates.
9. The TN-Qiskit timing is an implementation comparison between two classical evaluations of the same finite HHL map, not evidence of quantum advantage or superiority over optimized classical solvers.

## Referee 2 comments 3 and 4: memory and limited scaling

Run `python -m experiments.run_reviewer_r2_c3_c4` to regenerate these results. Qiskit exact-statevector and TN peak RSS are measured in separate spawned processes, without Qiskit sampling, and the reported increment is relative to each worker's post-warm-up RSS. The scaling study is empirical and limited to the declared finite ranges of `N` and `mu`; its log-log slopes are not asymptotic-complexity proofs. The separately timed spectral filter is used only as a numerical validation reference. Finally, the paired Qiskit-TN benchmark compares two classical simulations of the same finite HHL map, not quantum advantage and not performance against classical linear-system solvers.
