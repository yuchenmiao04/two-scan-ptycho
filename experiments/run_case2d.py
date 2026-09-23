"""Paired Poisson trials for the calibrated known-signature 2-D case."""
from __future__ import annotations

from experiments._shared import (
    ROOT, _manifest, _write_csv, _write_json, _number_summary, _support_string,
    _wilson, positive_integer,
)
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
from time import perf_counter
import numpy as np

from twoscan_ptycho.case2d import (
    add_poisson_noise_2d, build_calibrated_signature_model_2d,
    diffraction_intensity_2d, gaussian_probe_2d, perturbation_nrmse_2d,
    recover_calibrated_case_2d, sparse_phase_object_2d, support_exact,
)

CASE_N = 65
CASE_GAMMA = .015
CASE_BOUND = 10
CASE_LAGS = 12
CASE_SCANS = ((-2, -1), (3, 2))
CASE_SUPPORT = ((-9, -8), (0, 8), (8, -7))
CASE_PHASE = .8
TOTAL_PHOTONS = (6e7, 2e8, 6e8, 2e9, 6e9, 2e10)
_CASE_STATE = None

def _case_initialize():
    global _CASE_STATE
    signature = np.exp(1j * CASE_PHASE) - 1
    probe = gaussian_probe_2d(CASE_N, CASE_GAMMA)
    z, _ = sparse_phase_object_2d(CASE_N, CASE_SUPPORT, [CASE_PHASE] * 3)
    clean = tuple(diffraction_intensity_2d(z, probe, scan) for scan in CASE_SCANS)
    models = {
        "calibrated": build_calibrated_signature_model_2d(CASE_SCANS, CASE_GAMMA, CASE_LAGS, CASE_N, CASE_BOUND, signature, background_gamma=CASE_GAMMA),
        "gamma_plus_2pct": build_calibrated_signature_model_2d(CASE_SCANS, CASE_GAMMA * 1.02, CASE_LAGS, CASE_N, CASE_BOUND, signature, background_gamma=CASE_GAMMA),
    }
    _CASE_STATE = (signature, clean, models)


def _localization(truth, estimate):
    if len(truth) != len(estimate):
        return float("inf")
    return float(np.sqrt(min(sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(truth, ordered)) for ordered in permutations(estimate)) / len(truth)))


def _case_trial(task):
    total, trial, seed = task
    signature, clean, models = _CASE_STATE
    rng = np.random.default_rng(seed)
    noisy = tuple(add_poisson_noise_2d(frame, total / 2, rng) for frame in clean)
    rows = []
    for method, model in models.items():
        start = perf_counter()
        result = recover_calibrated_case_2d(noisy, model, 3, refinement_sweeps=3)
        runtime = perf_counter() - start
        rows.append({
            "total_photons_two_frames": total, "photons_per_frame": total / 2,
            "trial": trial, "seed": seed, "method": method,
            "support_exact": int(support_exact(CASE_SUPPORT, result.positions)),
            "localization_rmse_pixels": _localization(CASE_SUPPORT, result.positions),
            "perturbation_nrmse": perturbation_nrmse_2d(CASE_N, CASE_SUPPORT, [signature] * 3, result.positions, [signature * strength for strength in result.strengths]),
            "relative_residual": result.relative_residual,
            "estimated_positions": _support_string(result.positions),
            "estimated_strengths": _support_string(result.strengths),
            "runtime_seconds_parallel_diagnostic": runtime,
            "runtime_valid_for_method_comparison": 0,
        })
    return rows, np.asarray(noisy)


def run_case2d(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trials = 2 if args.smoke else args.trials
    budgets = (6e7, 2e9) if args.smoke else TOTAL_PHOTONS
    settings = {"n": CASE_N, "sparsity": 3, "gamma": CASE_GAMMA, "gamma_relative_error": .02, "positive_lag_box": [CASE_LAGS, CASE_LAGS], "candidate_grid": [-CASE_BOUND, CASE_BOUND], "scans": CASE_SCANS, "true_positions": CASE_SUPPORT, "known_common_signature_phase_rad": CASE_PHASE, "total_expected_photons_two_frames": budgets, "photons_per_frame": "total/2", "trials_per_budget": trials, "workers": args.workers, "refinement_sweeps": 3, "background_calibration": "exact blank-field autocorrelation at the true gamma for both reconstructions", "scope": "Calibrated common-signature nonnegative-strength case, not general complex-defect recovery"}
    _manifest(output, "case2d", args, settings)
    tasks = [(total, trial, args.seed + 30_000_041 + index * 100_003 + trial * 1_009) for total in budgets for index in (TOTAL_PHOTONS.index(total),) for trial in range(trials)]
    rows, measured_frames = [], []
    if args.workers == 1:
        _case_initialize()
        iterator = map(_case_trial, tasks)
        for j, (result, measurements) in enumerate(iterator, 1):
            rows.extend(result)
            measured_frames.append(measurements)
            if j % trials == 0:
                print(f"case2d budget {j // trials}/{len(budgets)} completed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_case_initialize) as executor:
            for j, (result, measurements) in enumerate(executor.map(_case_trial, tasks, chunksize=2), 1):
                rows.extend(result)
                measured_frames.append(measurements)
                if j % trials == 0:
                    print(f"case2d budget {j // trials}/{len(budgets)} completed", flush=True)
    # Archive the actual input arrays, not only regenerating seeds or estimates.
    np.savez_compressed(
        output / "measured_intensities.npz",
        intensities=np.asarray(measured_frames),
        total_photons_two_frames=np.asarray([task[0] for task in tasks]),
        trial=np.asarray([task[1] for task in tasks]),
        seeds=np.asarray([task[2] for task in tasks], dtype=np.int64),
        scan_centers=np.asarray(CASE_SCANS),
    )
    aggregate = []
    for total in budgets:
        for method in ("calibrated", "gamma_plus_2pct"):
            part = [row for row in rows if row["total_photons_two_frames"] == total and row["method"] == method]
            success = sum(row["support_exact"] for row in part)
            aggregate.append({"total_photons_two_frames": total, "photons_per_frame": total / 2, "method": method, "trials": len(part), "successes": success, "exact_support_rate": success / len(part), "wilson_95_interval": _wilson(success, len(part)), "localization_rmse_pixels": _number_summary([row["localization_rmse_pixels"] for row in part]), "perturbation_nrmse": _number_summary([row["perturbation_nrmse"] for row in part])})
    _write_csv(output / "trials.csv", rows)
    summary = {"settings": settings, "aggregate": aggregate, "runtime_warning": "Trial wall times are execution diagnostics, not isolated method timing measurements."}
    _write_json(output / "summary.json", summary)
    print(f"case2d: {len(rows)} paired reconstructions saved to {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/case2d")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--trials", type=positive_integer, default=200)
    parser.add_argument("--workers", type=positive_integer, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.output = args.output / "smoke"
    run_case2d(args)


if __name__ == "__main__":
    main()
