# Two-Scan Ptychography

Code and numerical results for **Sparse Ptychographic Defects: Two-Position
Recovery and Stability**.

Recover sparse complex defects on a known uniform background from two Gaussian
probe positions. This NumPy implementation includes spectral moment completion,
joint sparse least-squares fitting, four comparison methods, and the numerical
studies reported in the paper.

## Results at a glance

| Paper result | Experiment | Recorded results |
| --- | --- | --- |
| Fig. 1: Gaussian noise, `N = 1, 2, 3`, support recovery and defect NRMSE | `gaussian` | [gaussian](results/gaussian/) |
| Fig. 2(a): Poisson noise | `poisson` | [poisson](results/poisson/) |
| Fig. 2(b): Poisson plus read noise | `mixed` | [mixed](results/mixed/) |
| Fig. 2(c): defect separation | `separation` | [separation](results/separation/) |
| Fig. 2(d): two-dimensional known-signature case | `run_case2d` | [case2d](results/case2d/) |
| Noiseless algebraic and joint recovery | `run_validation` | [validation](results/validation/) |
| Background perturbation reported in the text | `background` | [background](results/background/) |
| Table I runtime column and online runtime ranges | `timing` | [timing](results/timing/) |
| Fresh setup plus reconstruction time | `run_timing_setup` | [timing_setup](results/timing_setup/) |
| Independent baseline parameter selection | `tune` | [baseline_tuning](results/baseline_tuning/) |

All accuracy tables retain incorrect-support outputs and numerical failures.
Recorded data are separate from rerun outputs in `results/reproduced/`.
The support rates and NRMSE in Table I come from the Gaussian comparison.

## Install and check

A CPU, Python and NumPy are sufficient; a GPU is not required.

```bash
git clone https://github.com/yuchenmiao04/two-scan-ptycho.git
cd two-scan-ptycho
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -e .
python -m experiments.verify_results
python -m unittest discover -s tests -v
python -m experiments.run_experiments smoke --workers 1
```

On Windows, activate with `.venv\Scripts\activate`. The smoke check uses
reduced budgets; it is not a paper result. The recorded comparison and timing
environment is CPython 3.10.11 with NumPy 2.2.6 on an Apple M4 (10-core CPU).

## Reproduce the comparisons

Run from the repository root. The selected baseline settings are loaded from
[frozen.json](results/baseline_tuning/frozen.json).

```bash
python -m experiments.run_experiments gaussian --workers 6
python -m experiments.run_experiments poisson --workers 6
python -m experiments.run_experiments mixed --workers 6
python -m experiments.run_experiments separation --workers 6
python -m experiments.run_experiments background --workers 6
```

Gaussian and photon-noise comparisons use 200 paired trials per condition;
separation and background-perturbation tests use 100. Within each paired trial,
all methods receive the same objects, measurements and specified prior
information. The methods are joint LS, full-data linearization with refinement,
sparse PIE, hard-thresholded Wirtinger flow and hard-thresholded amplitude flow.

To repeat independent baseline selection:

```bash
python -m experiments.run_experiments tune --workers 6
```

This evaluates 22 configurations on 20 objects at each of `N = 2, 3`, at
20 dB, 60 dB and without noise. Validation and evaluation use disjoint seeds.
To use newly selected settings, pass
`--frozen results/reproduced/baseline_tuning/frozen.json` to the comparison commands.
Full runs are substantially longer than the smoke check: iterative baselines
include up to 1,200 updates and 32 starts.

## Noiseless recovery and the two-dimensional case

These recorded studies used CPython 3.12.14 and NumPy 2.3.5:

```bash
python3.12 -m venv .venv-aux
.venv-aux/bin/python -m pip install -c constraints-auxiliary.txt -e .
.venv-aux/bin/python -m experiments.run_validation --trials 200
.venv-aux/bin/python -m experiments.run_case2d --trials 200 --workers 6
```

On Windows, use `.venv-aux\Scripts\python.exe`. Validation compares the joint
solver and spectral decoder on identical noiseless objects. The 2-D case estimates
nonnegative strengths of a known common phase-defect signature, with exact
calibration and a 2% probe-parameter error. It is a restricted model, not general
complex-valued 2-D recovery.

## Runtime measurements

Stop other experiments before running these serial benchmarks:

```bash
python -m experiments.run_experiments timing --workers 1
python -m experiments.run_timing_setup --trials 30 --confirm-isolated
```

Online timing includes all restarts and reconstruction, excluding input generation,
external scoring and file I/O. Joint online fitting reuses dictionary and support
factorizations; the fresh-setup experiment constructs them for each trial.
Both use 30 trials per condition and excluded warm-ups. The isolation flag is an
operator declaration, not automatic detection of competing workloads.

These are fixed-budget timings, not times to a common accuracy. Archived timings
are the paper's measurements; rerunning measures performance on your own machine.

## Conventions and scope

- Exact support means equality of the estimated and true integer-grid supports.
- Defect NRMSE is `norm(u_hat - u) / norm(u)`, excluding the known background
  from its normalization and including wrong-support estimates.
- Gaussian target SNR uses the defect-induced intensity `Y - Y_blank`.
- Photon budgets are expected counts across both frames, split equally. Mixed
  noise adds two-electron rms read noise per pixel.
- The theory assumes a calibrated Gaussian probe, a known uniform background,
  integer-grid defects, sufficient separation and a no-wrap margin.
- Joint LS enumerates admissible supports on small candidate grids. Its speed
  does not imply scalable enumeration or uniformly higher accuracy than
  full-data reconstruction.

See [the result inventory](results/README.md) for conditions, provenance and
integrity checks. Matching NumPy versions does not ensure bitwise identity across
CPU and BLAS implementations.

## Citation

```text
Y. Miao, M. Zhang, Z. Wang, R. Han, C. Han and S. Xu,
“Sparse Ptychographic Defects: Two-Position Recovery and Stability.”
```

Released under the [MIT License](LICENSE).
