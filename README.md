# Two-Scan Ptychography

Code and recorded numerical results for **Sparse Ptychographic Defects:
Two-Position Recovery and Stability**.

Recover sparse complex defects on a known uniform background from two Gaussian
probe positions. The NumPy implementation includes spectral moment completion,
joint sparse least-squares fitting, full-data linearization, sparse PIE,
hard-thresholded Wirtinger flow, and hard-thresholded amplitude flow.

## Paper results

| Result | Recorded data | Reproduction command |
| --- | --- | --- |
| Fig. 1: Gaussian support recovery and defect NRMSE, `N = 1, 2, 3` | [gaussian](results/gaussian/) | `python -m experiments.run_experiments gaussian --workers 6` |
| Fig. 2(a,b): Poisson and Poisson plus read noise | [poisson](results/poisson/), [mixed](results/mixed/) | `python -m experiments.run_experiments poisson --workers 6` and `mixed` |
| Fig. 2(c): two-dimensional known-signature case | [case2d](results/case2d/) | `python -m experiments.run_case2d --trials 200 --workers 6` |
| Fig. 2(d): defect separation | [separation](results/separation/) | `python -m experiments.run_experiments separation --workers 6` |
| Table I: Gaussian accuracy and support | [gaussian](results/gaussian/) | Gaussian command above |
| Table I: independent 80-dB serial timing | [timing_80db](results/timing_80db/) | `python -m experiments.run_timing_80db` |
| Table I: 60-dB serial timing | [timing](results/timing/) | `python -m experiments.run_experiments timing --workers 1` |
| Noiseless recovery and asymmetric-probe control | [validation](results/validation/) | `python -m experiments.run_validation --trials 200` |
| Background perturbation reported in the text | [background](results/background/) | `python -m experiments.run_experiments background --workers 6` |

At `N = 2` and 80 dB, joint LS and full-linear each recover all 200 supports in
the accuracy study; their median defect NRMSE values are `1.16e-4` and `2.07e-4`.
In a separate 30-pair serial benchmark, median cached online times are
`0.197 ms` and `15.258 ms`. The full-linear method has higher support reliability
at lower SNR and in the tested photon-noise conditions.

All accuracy summaries retain incorrect-support outputs and numerical failures.
The timing trials are separate from the 200-object accuracy trials. Recorded
data are preserved in `results/`; reruns write to `results/reproduced/`.

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

On Windows, activate with `.venv\Scripts\activate`. The smoke check uses reduced
budgets and is not a paper result. The recorded comparison and timing environment
is CPython 3.10.11 with NumPy 2.2.6 on an Apple M4 (10-core CPU).

The noiseless validation, conditioning and two-dimensional studies used CPython
3.12.14 and NumPy 2.3.5. Their recorded environment can be installed separately:

```bash
python3.12 -m venv .venv-aux
.venv-aux/bin/python -m pip install -c constraints-auxiliary.txt -e .
.venv-aux/bin/python -m experiments.run_validation --trials 200
.venv-aux/bin/python -m experiments.run_conditioning --trials 100
.venv-aux/bin/python -m experiments.run_case2d --trials 200 --workers 6
```

On Windows, use `.venv-aux\Scripts\python.exe`. Matching package versions does
not ensure bitwise identity across CPUs or BLAS implementations.

## Reproduce the comparisons

The selected baseline settings are loaded from
[frozen.json](results/baseline_tuning/frozen.json). Gaussian and photon-noise
comparisons use 200 paired trials per condition; separation and model-stress
comparisons use 100. All methods receive the same measurements and specified
prior information within a paired trial.

```bash
python -m experiments.run_experiments gaussian --workers 6
python -m experiments.run_experiments poisson --workers 6
python -m experiments.run_experiments mixed --workers 6
python -m experiments.run_experiments separation --workers 6
python -m experiments.run_experiments background --workers 6
```

To repeat independent baseline selection:

```bash
python -m experiments.run_experiments tune --workers 6
```

This evaluates 22 configurations on 20 objects at each of `N = 2, 3`, at 20 dB,
60 dB and without noise. Tuning and evaluation use disjoint seeds. Pass
`--frozen results/reproduced/baseline_tuning/frozen.json` to comparisons to use
newly selected settings. Full runs can take substantially longer than the smoke
check because the iterative baselines use up to 1,200 updates and 32 starts.

## Additional experiments

The complete model-stress study covers background and probe calibration,
position and gain error, probe truncation, quantization, separation, boundary
margin, off-grid defects, assumed sparsity, and weak defects.

```bash
python -m experiments.run_experiments model_stress --workers 6
python -m experiments.run_experiments beam_ablation --workers 6
python -m experiments.run_experiments unknown_sparsity --workers 6
python -m experiments.run_calibration blank --output results/reproduced/blank_calibration
python -m experiments.check_stability
```

The beam ablation uses a different search formulation from joint LS. Its failures
and diagnostic availability are retained. The known-center asymmetric-probe
control in validation tests a different single-position acquisition model.
The spectral decoder is an exact-arithmetic construction; finite-precision
failures are reported rather than treated as missing observations. The 2-D
case estimates nonnegative strengths of a known common phase-defect signature.

## Runtime measurements

Stop competing experiments before running the serial benchmarks:

```bash
python -m experiments.run_experiments timing --workers 1
python -m experiments.run_timing_80db
python -m experiments.run_timing_setup --trials 30 --confirm-isolated
```

Online timing includes lag extraction and reconstruction, or every baseline's
prescribed optimization and restarts. Joint online fitting reuses dictionary
and support factorizations; the fresh-setup study constructs them for each
trial. Input generation, external scoring and file I/O are excluded. Timing
uses 30 trials per condition with excluded warm-ups. The isolation flag is an
operator declaration, not automatic detection of competing workloads.

These fixed-budget timings are not times to a common accuracy. The recorded
values are the paper's measurements; a rerun measures the local machine.

## Recreate the figures

Plotting is optional and does not rerun a solver:

```bash
python -m pip install -e '.[plot]'
python -m experiments.plot_figures
```

This verifies the recorded data and recreates both manuscript figures under
`results/reproduced/figures/`. Add `--formats png svg pdf` for a PDF copy.
The [figure previews](figures/) are generated from the same per-trial records.

## Conventions and scope

- Exact support means equality of the estimated and true integer-grid supports.
- Defect NRMSE is `norm(u_hat - u) / norm(u)`, including wrong-support estimates.
- Gaussian target SNR uses the defect-induced intensity `Y - Y_blank`.
- Photon budgets are expected counts across both frames, split equally. Mixed
  noise adds two-electron rms read noise per pixel.
- The theory assumes a calibrated Gaussian probe, a known uniform background,
  integer-grid defects, sufficient separation and a no-wrap margin.
- Joint LS enumerates admissible supports on small candidate grids; the measured
  speed does not establish scalable enumeration or uniformly higher accuracy.

See the [result inventory](results/README.md) for conditions, provenance and
integrity checks. Public records retain numerical outcomes, seeds and technical
environments while omitting calendar timestamps and personal machine identifiers.

## Citation

```text
Y. Miao, M. Zhang, Z. Wang, R. Han, C. Han and S. Xu,
“Sparse Ptychographic Defects: Two-Position Recovery and Stability.”
```

Released under the [MIT License](LICENSE).
