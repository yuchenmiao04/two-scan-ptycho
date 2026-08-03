# Frozen result files

These files are the frozen numerical outputs used by the paper.

| Directory | Contents |
| --- | --- |
| `validation/` | deterministic Fig. 1(a) trace and ten noiseless trials for each `N=1,...,4` |
| `separation/` | trial records and aggregate rates for Fig. 1(b) |
| `noise/` | 200-trial clean-lag SNR sweep for Fig. 2(a) |
| `comparison/` | paired proposed/full-linear/PIE/HT-WF records for Fig. 2(b) |
| `poisson_2d/` | photon sweep, fixed Poisson realization, and support-fit array for Fig. 3 |

Every JSON summary records the seed and configuration. CSV files contain one row per
trial. `fixed_case.npz` contains the fixed measured frames and arrays needed to inspect
the reported two-dimensional localization case.
