# Recorded numerical results

These directories contain the numerical evidence for **Sparse Ptychographic
Defects: Two-Position Recovery and Stability** and the accompanying experimental
studies. Trial CSVs preserve numerical values, seeds, incorrect supports and
numerical failures. Undefined metrics remain empty rather than being replaced
with zero. Summaries and data hashes can be checked without running a solver.

| Directory | Experiment | Method-trial rows |
| --- | --- | ---: |
| `baseline_tuning` | Independent parameter selection: 22 configurations, 20 objects at each of six sparsity/SNR conditions | 2,640 |
| `tuning_reproduction` | Independent replay of the parameter-selection study | 2,640 |
| `gaussian` | Three sparsities, seven finite SNR levels and a noiseless control, 200 paired trials | 24,000 |
| `poisson` | Two-defect Poisson comparison, seven photon budgets, 200 paired trials | 7,000 |
| `mixed` | The same photon budgets with two-electron rms read noise | 7,000 |
| `separation` | Spacings 2, 4, 5, 6, 8 at 60 dB, 100 paired trials and five methods | 2,500 |
| `background` | Background perturbations 0 and 0.001 at 60 dB, 100 paired trials and five methods | 1,000 |
| `model_stress` | Complete 53-condition model-stress design, 100 paired trials and five methods | 26,500 |
| `timing` | Three sparsities, 20/60/noiseless dB, 30 serial paired trials and five methods | 1,350 |
| `timing_80db` | Separate `N = 2`, 80-dB serial benchmark, 30 paired trials and two methods | 60 |
| `timing_setup` | Fresh joint setup plus reconstruction, 30 trials per sparsity at 60 dB | 90 |
| `validation` | Joint, spectral and known-center asymmetric-probe validation, 200 objects per sparsity | 1,800 |
| `case2d` | Known-signature 2-D case, six photon budgets, exact calibration and 2% probe-width mismatch | 2,400 |
| `conditioning` | Eight sparsities, five probe widths, 100 support pairs per condition | 4,000 |
| `blank_calibration` | Independent noisy blank exposures, three object SNRs and five calibration settings | 3,000 |
| `beam_ablation` | Five beam widths, four noise settings, 100 paired trials and two methods | 4,000 |
| `unknown_sparsity` | True count two with upper bound three, five noise settings, 200 trials | 1,000 |
| `stability_check` | Bound and localization checks on the blank-calibration observations | 3,000 |

The 93,980 rows include related views of the same observations: `separation` and
`background` are subsets of `model_stress`, while `stability_check` analyzes the
blank-calibration trials. They should not be counted as independent experiments.

```bash
python -m experiments.verify_results
```

This read-only command checks file hashes, trial identities, condition coverage,
summary statistics, retained measurement arrays and the numerical values used
in the manuscript. It does not substitute rerun results for the recorded data.

## Reproduction and provenance

The paired comparison, tuning and timing studies used CPython 3.10.11 with NumPy
2.2.6. Noiseless validation, conditioning and the two-dimensional case used
CPython 3.12.14 with NumPy 2.3.5. Technical environment details are retained in the
manifests. The serial benchmarks used an Apple M4 with one requested BLAS thread.
Hardware and numerical libraries affect timings and floating-point output.

Exact job descriptions preserve trial indices, seed namespaces and solver
settings. Separation retains the `cluster` seed tag, background perturbation
retains the `background` tag, and model-stress records retain their acquisition
suite identifier `mismatch`. The public directory names do not change pairing.

`baseline_tuning/frozen.json` contains the selected settings and tuning scores.
`recorded_sources` retains source hashes recorded with the original measurements
under neutral component identifiers. These historical values are not claimed
to hash the distributed files. `distribution_source_hashes` separately
identifies the code supplied here. Numerical records are preserved; packaging,
module names and metadata have been organized for distribution.

Calendar timestamps, personal account paths, host names and machine identifiers
are omitted. Scientific seeds, trial indices, software versions and performance
durations are retained.
Validation arrays contain measured frames, supports and complex amplitudes,
including the known-center asymmetric-probe control. The 2-D array records
contain frames identified by photon budget, trial and seed, shared by both
calibration models. Conditioning arrays retain dictionaries, singular values
and the complete two-defect union-gain check. Other measurements can be
regenerated from their specified seeds and acquisition models. To recompute the
stability analysis from measurement arrays, first run the blank-calibration
experiment into `results/reproduced/blank_calibration`, then run
`python -m experiments.check_stability`.

## Metrics and experimental scope

Defect NRMSE is `norm(u_hat-u)/norm(u)`, including wrong-support outputs. Gaussian
SNR uses the defect-induced intensity, excluding the known blank field from its
normalization. Photon budgets are total expected counts across the two frames,
split equally. Support intervals are pointwise 95% Wilson intervals.

All methods receive identical measurements within a paired trial. Gaussian SNR
levels share objects and standardized noise directions. Poisson and mixed-noise
studies share objects and expected photon budgets, but their shot-noise frames
are not identical because read-noise sampling advances the random stream.

The separation study removes the separation restriction from every method's
search class so that the true support remains admissible. Background
perturbation is a mismatch in the object model, distinct from acquiring a noisy
blank exposure. The two-dimensional study uses a known common phase-defect
signature and estimates nonnegative strengths, not arbitrary complex 2-D
coefficients.

Finite-precision failures of the algebraic spectral decoder remain in the
validation data. In particular, its `N = 3` support success is 29/200; the global
joint fit recovers all 600 tested noiseless supports. The asymmetric-probe control
has a known noninteger probe center and tests a different acquisition geometry.
The beam study uses a different search class from joint fitting; unavailable
internal beam diagnostics remain unavailable and are not assigned zero error.

## Timing scope

Table I accuracy comes from the 200-object Gaussian study. Its 80-dB timings
come from the separate 30-pair serial study in `timing_80db`. Both methods recover
30/30 supports there; this does not replace the 200-object accuracy sample.
Median cached online times are 0.1971459 ms for joint LS and 15.2583955 ms for
full-linear. The median of paired speedup ratios is 77.7, distinct from the ratio
of the two timing medians.

Online timing includes joint lag extraction and cached support fits, or each
baseline's optimization and internal diagnostics, including restarts. It
excludes input generation, external scoring and file I/O. Cached setup fields
are not repeated fresh-setup measurements. `timing_setup` builds a new joint
dictionary and support factorization for each retained trial, with excluded
warm-ups; it does not flush operating-system caches or include process startup.
Parallel accuracy-run wall times are diagnostic values. These fixed-budget
benchmarks do not measure time to a common accuracy.
