# Recorded numerical results

The directories below contain the numerical experiments for **Sparse
Ptychographic Defects: Two-Position Recovery and Stability**. Each `trials.csv`
contains every recorded method trial for its stated conditions, including
incorrect supports and numerical failures. `summary.json` gives condition-level
statistics recomputed from those rows. `manifest.json` records the experiment
design, seeds, acquisition environment, and file hashes.

| Directory | Manuscript experiment | Method-trial rows |
| --- | --- | ---: |
| `baseline_tuning` | Independent baseline selection; 20 objects at each of six sparsity/SNR conditions, 22 configurations | 2,640 |
| `gaussian` | Gaussian comparison; three sparsities, eight SNR levels, 200 paired trials | 24,000 |
| `poisson` | Two-defect Poisson comparison; seven photon budgets, 200 paired trials | 7,000 |
| `mixed` | Same budgets with two-electron rms read noise | 7,000 |
| `separation` | Spacings 2, 4, 5, 6, 8 at 60 dB; 100 paired trials and all five methods | 2,500 |
| `background` | Background perturbations 0 and 0.001 at 60 dB; 100 paired trials and all five methods | 1,000 |
| `timing` | Serial online timing; three sparsities, 20/60/noiseless dB, 30 paired trials and all five methods | 1,350 |
| `timing_setup` | Fresh joint setup plus reconstruction; 30 trials for each sparsity at 60 dB | 90 |
| `validation` | Separate noiseless joint and spectral validation; 200 objects for each sparsity | 1,200 |
| `case2d` | Known-signature 2-D case; six photon budgets, exact calibration and 2% error in gamma, 200 paired trials | 2,400 |

Run the read-only consistency checks from the repository root:

```bash
python -m experiments.verify_results
```

## Reproduction and provenance

The paired comparison, tuning, and timing data were computed on native-arm64
macOS 15.5 with CPython 3.10.11 and NumPy 2.2.6. The noiseless validation and 2-D
case used CPython 3.12.14 and NumPy 2.3.5 on the same Apple M4 host (10 CPU cores,
24 GiB memory). Recorded timing depends on the host and numerical libraries;
matching package versions does not guarantee identical floating-point results
on another CPU or BLAS implementation.

Tuning uses seed namespace `19692001`, accuracy comparisons use `20260920`, and
timing uses `19371011`. The exact job descriptions preserve each trial index,
namespace and solver settings. Object and noise seeds depend on the condition
type as well as the trial: separation retains the `cluster` seed tag, and the
background study retains the `background` tag. Those studies must not reuse the
ordinary Gaussian control objects. Validation and 2-D rows contain their actual
integer seeds. The validation arrays retain the measured frames, supports and
complex defect amplitudes. The 2-D arrays retain the measured frames; its
manifest records the true support and common defect signature. Each 2-D array
record is identified by photon budget, trial and seed, and is shared by both
reconstruction models.

`baseline_tuning/frozen.json` contains the selected settings and every tuning
configuration's score. `recorded_sources` and `recorded_solver_sources` retain
the acquisition-time SHA-256 values under neutral component identifiers. They
are not claimed to hash the currently distributed files.
`distribution_source_hashes` separately authenticates the distributed code;
the runner checks its solver entries before using the frozen settings. Public
CSV numerical cells are the recorded cells, not newly simulated replacements.

## Metrics and experimental scope

Defect NRMSE is `norm(u_hat-u)/norm(u)`, including wrong-support outputs.
Gaussian SNR is defined using defect-induced intensity, not total intensity
including the known background. Photon budgets are the total expected counts
across both frames, divided equally between the frames. Support intervals are
95% Wilson intervals. Undefined metrics remain empty in CSV rather than being
silently replaced with zero.

The separation study removes the support-separation restriction for every
method, so all true supports remain admissible. The background study adds
`delta * (cos(2*pi*q/n) + 0.5j*sin(4*pi*q/n))` to the object background while
reconstruction assumes the nominal calibrated background. Its two perturbation
levels share the same defect realization and standardized noise direction.
This is a background-model sensitivity experiment, not an additional noisy
blank-exposure experiment.

Within a trial, all methods receive identical measurements. Gaussian SNR
levels share objects and standardized noise directions. Poisson and mixed-noise
studies share objects and expected photon budgets, but their Poisson frames are
not identical because read-noise sampling advances the random stream.

Online timing includes joint lag extraction and cached support fits, or each
baseline's optimization and internal diagnostics, including all restarts. It
excludes input generation, external scoring and file I/O. `timing` uses one
process and one BLAS thread, after an excluded warm-up. Its cached setup values
are not repeated fresh-setup measurements. `timing_setup` builds a fresh joint
dictionary and support factorization for every retained trial, with an excluded
warm-up per sparsity; it does not flush operating-system caches or include
process startup. These fixed-budget timings do not measure time to a common
accuracy. Parallel accuracy-run wall times are diagnostics, not controlled
method-timing comparisons.
