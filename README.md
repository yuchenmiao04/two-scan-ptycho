# Two-Scan Ptychography

Reference implementation and frozen numerical results for **“Two-Scan Recovery of
Sparse Defects in Ptychography.”**

The code reconstructs sparse defects on a known uniform background from two square-DFT
intensity measurements. It implements the paper's clean-lag moment decoder, the
full-data linearization baseline, PIE, hard-thresholded Wirtinger flow, and the
two-dimensional calibrated Poisson case.

Each entry point below recreates one reported experiment and writes its numerical
outputs to `results/`.

## What is included

| Paper result | Entry point | Frozen output |
| --- | --- | --- |
| Fig. 1(a), clean-lag identity | `experiments/run_clean_lag_trace.py` | `results/validation/clean_lag_trace.*` |
| Noiseless exactness reported in Sec. V-A | `experiments/run_validation.py` | `results/validation/summary.json` |
| Fig. 1(b), strict separation transition | `experiments/run_separation.py` | `results/separation/` |
| Fig. 2(a), proposed decoder for `N=1,2,3` | `experiments/run_noise.py` | `results/noise/` |
| Fig. 2(b), paired four-method comparison | `experiments/run_comparison.py` | `results/comparison/` |
| Fig. 3, two-dimensional Poisson case | `experiments/run_poisson_2d.py` | `results/poisson_2d/` |

## Installation

Python 3.10 or newer and NumPy are sufficient.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quick check

The deterministic trace and unit tests run quickly on a CPU:

```bash
python -m experiments.run_clean_lag_trace
python -m unittest discover -s tests -v
```

The first command regenerates the numerical values underlying Fig. 1(a).

## Reproduce the paper results

Run each command from the repository root. The defaults are the exact configurations
used in the paper and overwrite the corresponding frozen result files.

### Noiseless algebra and exact reconstruction

```bash
python -m experiments.run_validation
```

This performs ten trials for each `N=1,...,4`, checks the clean-lag identity and the
one-scan reflection ambiguity, and runs the finite two-scan decoder.

### Strict separation boundary

```bash
python -m experiments.run_separation
```

This evaluates `N=2,3` on 100 trials at every spacing shown in Fig. 1(b).

### Noise robustness for `N=1,2,3`

```bash
python -m experiments.run_noise
```

This runs 200 trials per SNR and sparsity using the minimum `2N` clean lags.

### Paired baseline comparison

```bash
python -m experiments.run_comparison
```

All four methods receive the same 50 objects and noise directions at every SNR. The
configuration is:

- proposed clean-lag decoder: beam width 256;
- full-data linearization: four quadratic debiasing updates;
- PIE: 800 iterations and 32 starts;
- HT-WF: 1,200 iterations and 32 starts.

This is the longest one-dimensional experiment because the iterative baselines use
multiple starts.

### Two-dimensional Poisson case

```bash
python -m experiments.run_poisson_2d
```

This reproduces the fixed `10^9`-photon localization and the six photon-budget points
in Fig. 3. The result archive stores the measured Poisson frames and profiled support
score used in panel (a).

## Repository layout

```text
two-scan-ptycho/
├── experiments/          # one entry point per reported experiment
├── results/              # frozen JSON, CSV, and NPZ outputs
├── src/twoscan_ptycho/
│   ├── core.py           # clean-lag decoder and orbit integration
│   ├── full_data.py      # full-data linearization baseline
│   ├── iterative.py      # PIE and hard-thresholded WF
│   └── case2d.py         # calibrated 2-D Poisson decoder
└── tests/                # deterministic correctness checks
```

## Metrics and conventions

- Arrays use centered coordinates `[-M, ..., M]` for odd `n=2M+1`.
- The implementation uses NumPy's ordinary FFT/IFFT normalization.
- Support success means exact equality of recovered and true integer supports.
- NRMSE is evaluated on the sparse perturbation, not the known unit background.
- Gaussian-noise SNR is measured relative to the complete intensity vector.
- Runtime fields are wall-clock seconds and include all restarts and support fitting.

## Frozen headline results

- The noiseless decoder recovered all ten trials for every `N=1,...,4`.
- The separation sweep succeeded in every trial for `Delta > 2N` and failed at the
  boundary used in Fig. 1(b).
- In the paired `N=2` comparison, the proposed method reached 86% exact support at
  90 dB and 100% at 110 dB.
- The calibrated two-dimensional case reached 80% exact support at `10^9` photons and
  100% at `3x10^9` photons.

See [`results/README.md`](results/README.md) for the file-level provenance.

## Citation

If this code is useful, please cite the accompanying paper:

```text
Y. Miao, M. Zhang, Z. Wang, R. Han, C. Han and S. Xu,
“Two-Scan Recovery of Sparse Defects in Ptychography.”
```

The implementation is released under the [MIT License](LICENSE).
