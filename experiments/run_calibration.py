"""Independent blank-field calibration and cold-setup studies for sparse-defect reconstruction.

This entry point does not modify the production experiment runner or its source
hash manifest. Blank-field photon noise affects the estimated autocorrelation
that is subtracted, not the already generated object diffraction measurements.
"""
from __future__ import annotations

import os

for _variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_experiments import (
    SEEDS, calculate_metrics, configuration, dump_json, hardware, make_instance,
    source_hashes as production_source_hashes, stable_seed, wilson,
)
from twoscan_ptycho.reconstruction import (
    GlobalSupportSolver, build_joint_dictionary, extract_raw_data,
)


def calibrate_blank(blank_intensities, total_photons, rng):
    """Acquire independent blank frames using a known, fixed exposure gain.

    Each frame receives total_photons / number_of_frames expected detected
    photons. gain_i = photons_per_frame / sum_j I_blank[i,j] is fixed before
    sampling, and I_hat[i,j] = counts[i,j] / gain_i. In particular, random
    observed count totals are never used to renormalize the estimate.
    """
    blank = np.asarray(blank_intensities, dtype=float)
    if blank.ndim != 2 or not np.all(np.isfinite(blank)) or np.any(blank < 0):
        raise ValueError("blank_intensities must be a finite nonnegative frame matrix")
    if not np.all(blank.sum(axis=1) > 0):
        raise ValueError("Every blank frame must have positive total intensity")
    if total_photons <= 0 or not np.isfinite(total_photons):
        raise ValueError("total_photons must be finite and positive")
    photons_per_frame = float(total_photons) / len(blank)
    gain = photons_per_frame / blank.sum(axis=1)
    counts = rng.poisson(blank * gain[:, None])
    estimate = counts / gain[:, None]
    return {
        "counts": counts,
        "intensities": estimate,
        "autocorrelations": np.fft.ifft(estimate, axis=1),
        "exposure_gain": gain,
        "total_expected_photons": float(total_photons),
        "photons_per_frame": photons_per_frame,
    }


def raw_lags_with_blank_autocorrelation(intensities, dictionary, estimated_cw):
    """Subtract estimated Cw from IFFT(Y), without changing Y or the dictionary."""
    frames = np.asarray(intensities, dtype=float)
    cw = np.asarray(estimated_cw, dtype=complex)
    shape = (len(dictionary.scans), dictionary.n)
    if frames.shape != shape or cw.shape != shape:
        raise ValueError("Measured frames and blank autocorrelations must match the dictionary")
    if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(cw)):
        raise ValueError("Measured frames and blank autocorrelations must be finite")
    lag_data = (np.fft.ifft(frames, axis=1) - cw)[:, 1:dictionary.lag_count + 1]
    return np.concatenate([part for frame in lag_data for part in (frame.real, frame.imag)])


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def distribution(values):
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": len(finite), "mean": float(np.mean(finite)),
        "median": float(np.median(finite)), "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "q25_q75": np.quantile(finite, [.25, .75]).tolist(),
        "q10_q90": np.quantile(finite, [.1, .9]).tolist(),
    }


def prepare_output(output, settings):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite or mix an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    hashes = production_source_hashes()
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    dump_json(output / "manifest.json", {
        "settings": settings, "hardware": hardware(), "workers": 1,
        "source_hashes": hashes,
        "note": "Independent runner; production source_hashes and production results are unchanged.",
    })
    return output


def write_rows(output, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fresh_solver(sparsity):
    """Build directly, without the production runner's memoized joint_solver."""
    bound, gamma = configuration(sparsity)
    dictionary = build_joint_dictionary(
        257, gamma, (0, 1), tuple(range(-bound, bound + 1)), sparsity,
        minimum_separation=2 * sparsity, enforce_margin=True,
    )
    return GlobalSupportSolver(dictionary, sparsity)


def run_blank(args):
    settings = {
        "scope": "joint_only_blank_calibration",
        "sparsity": 2, "n": 257, "positions": [0, 1], "candidate_bound": 10,
        "gamma": .05, "trials_per_condition": args.trials,
        "namespace": args.namespace, "trial_start": args.trial_start,
        "object_generation": "run_experiments.make_instance, suite=gaussian, stress=none",
        "object_effective_snr_db": args.snrs,
        "calibration_total_expected_photons": args.calibration_photons,
        "calibration_frames": 2, "calibration_photons_per_frame": [
            value / 2 for value in args.calibration_photons
        ],
        "calibration_model": "Independent Poisson blank exposure at each position; independent of object noise.",
        "known_exposure_gain": "g_i=(P_total/2)/sum(I_blank_i), fixed before sampling; I_hat_i=counts_i/g_i",
        "observed_count_total_renormalization": False,
        "pairing": "Same truth and measured Y across all blank budgets; same blank realization across SNRs.",
        "blank_seed": "stable_seed(namespace,'blank-calibration',trial,total_photons)",
        "exact_control": "Ideal exact Cw reference, zero calibration error, no finite calibration photon cost assigned.",
        "calibration_acquisition_cost": "P_total is additional blank-exposure photons; not object photons or absorbed dose.",
        "solver": "Global raw-lag least squares over the same legal supports; no baseline comparison.",
        "cache": "One joint solver reused; setup excluded from diagnostic online times.",
        "runtime_scope": "Execution diagnostics only, not isolated comparative timing.",
        "theory_scope": "No interval certificate or numerically certified rho/beta bound is claimed.",
    }
    output = prepare_output(args.output, settings)
    measurements, calibrations = output / "measurements", output / "calibrations"
    measurements.mkdir()
    calibrations.mkdir()
    started = time.perf_counter()
    solver = fresh_solver(2)
    model = solver.dictionary
    exact_cw = np.fft.ifft(model.background_intensities, axis=1)
    rows = []
    for trial in range(args.trial_start, args.trial_start + args.trials):
        blank_estimates = []
        for budget in args.calibration_photons:
            seed = stable_seed(args.namespace, "blank-calibration", trial, budget)
            estimate = calibrate_blank(
                model.background_intensities, budget, np.random.default_rng(seed)
            )
            name = f"trial{trial:05d}_photons{budget:g}.npz"
            np.savez_compressed(
                calibrations / name, counts=estimate["counts"],
                estimated_intensities=estimate["intensities"],
                estimated_cw=estimate["autocorrelations"],
                exact_intensities=model.background_intensities,
                exposure_gain=estimate["exposure_gain"], seed=np.uint64(seed),
            )
            blank_estimates.append((budget, seed, estimate))
        for snr in args.snrs:
            instance = make_instance({
                "suite": "gaussian", "namespace": args.namespace,
                "trial": trial, "sparsity": 2, "snr": snr,
            })
            measured = instance["measured"]
            measurement_hash = array_hash(measured)
            np.savez_compressed(
                measurements / f"trial{trial:05d}_snr{snr}.npz",
                measured=measured, noiseless=instance["clean"], truth=instance["truth"],
                positions=instance["positions"], amplitudes=instance["amplitudes"],
            )
            clean_b = raw_lags_with_blank_autocorrelation(instance["clean"], model, exact_cw)
            exact_b = raw_lags_with_blank_autocorrelation(measured, model, exact_cw)
            measurement_noise = float(np.linalg.norm(exact_b - clean_b))
            api_difference = float(np.linalg.norm(exact_b - extract_raw_data(measured, model)))
            choices = [(None, None, None)] + blank_estimates
            for budget, seed, blank in choices:
                cw = exact_cw if blank is None else blank["autocorrelations"]
                start_online = time.perf_counter()
                b = raw_lags_with_blank_autocorrelation(measured, model, cw)
                row = {
                    "suite": "blank", "method": "joint", "trial": trial,
                    "sparsity": 2, "snr": snr,
                    "calibration": "exact" if blank is None else "poisson",
                    "calibration_total_photons": budget,
                    "calibration_photons_per_frame": None if budget is None else budget / 2,
                    "blank_seed": seed, "truth_sha256": array_hash(instance["truth"]),
                    "measurement_sha256": measurement_hash,
                    "true_support": ";".join(map(str, instance["positions"])),
                    "effective_snr_db": instance["effective_snr"],
                    "total_snr_db": instance["total_snr"], "sigma": instance["sigma"],
                    "blank_lag_error_norm": float(np.linalg.norm(b - exact_b)),
                    "measurement_raw_lag_noise_norm": measurement_noise,
                    "combined_raw_lag_error_norm": float(np.linalg.norm(b - clean_b)),
                    "exact_control_existing_api_difference": api_difference if blank is None else None,
                    "coefficient_min": float(np.min(np.abs(instance["amplitudes"]))),
                    "legal_support_count": solver.candidate_count,
                }
                try:
                    answer = solver.solve(b)
                    online = time.perf_counter() - start_online
                    row.update(calculate_metrics(instance, answer.u, answer.support))
                    row.update(
                        status=answer.status, failed=0, raw_lag_residual=answer.residual,
                        residual_selection_slack=answer.diagnostics["residual_selection_slack"],
                        rank_dropped_support_count=solver.rank_dropped_count,
                        online_seconds=online,
                    )
                except Exception as error:
                    row.update(
                        status=f"{type(error).__name__}: {error}", failed=1,
                        support_exact=0, defect_nrmse=None,
                        online_seconds=time.perf_counter() - start_online,
                    )
                rows.append(row)
            if array_hash(measured) != measurement_hash:
                raise RuntimeError("Object measurements were mutated during blank calibration")
    groups = {}
    for row in rows:
        key = (row["snr"], row["calibration"], row["calibration_total_photons"])
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (snr, calibration, budget), group in groups.items():
        successes = sum(row["support_exact"] for row in group)
        item = {
            "snr": snr, "calibration": calibration, "calibration_total_photons": budget,
            "trials": len(group), "successes": successes,
            "exact_support_rate": successes / len(group),
            "wilson_95_interval": wilson(successes, len(group)),
            "status_counts": dict(Counter(row["status"] for row in group)),
            "failures": sum(row["failed"] for row in group),
        }
        for field in (
            "blank_lag_error_norm", "measurement_raw_lag_noise_norm",
            "combined_raw_lag_error_norm", "defect_nrmse", "raw_lag_residual",
            "online_seconds", "localization_rmse", "amplitude_relative_error",
            "phase_rmse_rad",
        ):
            item[field] = distribution([row.get(field) for row in group])
        aggregate.append(item)
    write_rows(output, rows)
    dump_json(output / "summary.json", {
        "settings": settings, "rows": len(rows), "aggregate": aggregate,
        "batch_elapsed_seconds": time.perf_counter() - started,
        "runtime_warning": settings["runtime_scope"],
    })
    print(f"blank: {len(rows)} paired reconstructions written to {output}", flush=True)
    return rows


def cold_setup_and_online(sparsity, measured):
    """Time actual new objects, not retrieval of a memoized support solver.

    The online interval reuses only the SVD data of this freshly constructed
    solver. A previous repetition's dictionary or solver is never retrieved.
    Object generation, metric calculation and disk I/O are outside both timers.
    """
    bound, gamma = configuration(sparsity)
    candidates = tuple(range(-bound, bound + 1))
    start = time.perf_counter()
    dictionary = build_joint_dictionary(
        257, gamma, (0, 1), candidates, sparsity,
        minimum_separation=2 * sparsity, enforce_margin=True,
    )
    dictionary_end = time.perf_counter()
    solver = GlobalSupportSolver(dictionary, sparsity)
    setup_end = time.perf_counter()
    answer = solver.solve_intensities(measured)
    online_end = time.perf_counter()
    timing = {
        "dictionary_build_seconds": dictionary_end - start,
        "solver_precompute_seconds": setup_end - dictionary_end,
        "cold_setup_seconds": setup_end - start,
        "cached_online_seconds": online_end - setup_end,
        "total_cold_reconstruction_seconds": online_end - start,
        "candidate_sites": len(candidates),
        "legal_support_count": solver.candidate_count,
        "rank_dropped_support_count": solver.rank_dropped_count,
        "status": answer.status,
        "raw_lag_residual": answer.residual,
    }
    return answer, timing


def run_timing_setup(args):
    settings = {
        "scope": "joint_dictionary_and_global_solver_cold_setup",
        "sparsities": args.sparsities, "repetitions_per_sparsity": args.trials,
        "namespace": args.namespace, "trial_start": args.trial_start,
        "object_effective_snr_db": args.snr,
        "setup_cache_disabled": True, "warmup_per_sparsity": 1,
        "online_scope": "IFFT/background subtraction plus solve_intensities on the newly built solver; its SVD factors are reused.",
        "cold_setup_scope": "Direct build_joint_dictionary plus GlobalSupportSolver construction, including legal-support enumeration and SVD.",
        "excluded": "Object/noise generation, file I/O, metric calculation, and one warmup per sparsity.",
        "cold_definition": "Fresh dictionary and solver every repetition; not an OS cache flush or new-process startup.",
        "concurrency_isolation_user_confirmed": args.confirm_isolated,
        "workers": 1, "blas_threads_requested": 1,
        "warmup_namespace": "stable_seed(namespace,'setup-warmup')",
    }
    output = prepare_output(args.output, settings)
    rows = []
    started = time.perf_counter()
    for sparsity in args.sparsities:
        warmup = make_instance({
            "suite": "gaussian", "namespace": stable_seed(args.namespace, "setup-warmup"),
            "trial": 0, "sparsity": sparsity, "snr": args.snr,
        })
        cold_setup_and_online(sparsity, warmup["measured"])
        for trial in range(args.trial_start, args.trial_start + args.trials):
            instance = make_instance({
                "suite": "gaussian", "namespace": args.namespace,
                "trial": trial, "sparsity": sparsity, "snr": args.snr,
            })
            answer, timing = cold_setup_and_online(sparsity, instance["measured"])
            row = {
                "suite": "timing_setup", "method": "joint", "trial": trial,
                "sparsity": sparsity, "snr": args.snr,
                "truth_sha256": array_hash(instance["truth"]),
                "measurement_sha256": array_hash(instance["measured"]),
                **timing,
            }
            row.update(calculate_metrics(instance, answer.u, answer.support))
            rows.append(row)
    aggregate = []
    for sparsity in args.sparsities:
        group = [row for row in rows if row["sparsity"] == sparsity]
        item = {
            "sparsity": sparsity, "trials": len(group),
            "status_counts": dict(Counter(row["status"] for row in group)),
            "support_success_rate": sum(row["support_exact"] for row in group) / len(group),
        }
        for field in (
            "dictionary_build_seconds", "solver_precompute_seconds",
            "cold_setup_seconds", "cached_online_seconds",
            "total_cold_reconstruction_seconds",
        ):
            item[field] = distribution([row[field] for row in group])
        aggregate.append(item)
    write_rows(output, rows)
    dump_json(output / "summary.json", {
        "settings": settings, "rows": len(rows), "aggregate": aggregate,
        "batch_elapsed_seconds": time.perf_counter() - started,
        "runtime_warning": (
            "Single worker and BLAS threads requested; system isolation was declared by the operator, not verified by the runner."
            if args.confirm_isolated else
            "Isolation not confirmed. These timings are diagnostics and must not be used as isolated performance results."
        ),
    })
    print(f"timing_setup: {len(rows)} fresh reconstructions written to {output}", flush=True)
    return rows


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if number <= 0 or not np.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    blank = subparsers.add_parser("blank", help="Joint-only independent Poisson blank-calibration sweep")
    blank.add_argument("--output", required=True, type=Path)
    blank.add_argument("--trials", type=positive_integer, default=200)
    blank.add_argument("--trial-start", type=int, default=0)
    blank.add_argument("--namespace", type=int, default=SEEDS["formal"])
    blank.add_argument("--snrs", nargs="+", choices=("40", "60", "inf"), default=["40", "60", "inf"])
    blank.add_argument("--calibration-photons", nargs="+", type=positive_float,
                       default=[1e6, 1e8, 1e10, 1e12])
    blank.add_argument("--workers", type=int, choices=(1,), default=1)
    blank.set_defaults(run=run_blank)
    timing = subparsers.add_parser(
        "timing_setup", help="Serial fresh dictionary/SVD setup and separately timed online reconstruction"
    )
    timing.add_argument("--output", required=True, type=Path)
    timing.add_argument("--trials", type=positive_integer, default=30)
    timing.add_argument("--trial-start", type=int, default=0)
    timing.add_argument("--namespace", type=int, default=SEEDS["timing"])
    timing.add_argument("--sparsities", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    timing.add_argument("--snr", choices=("40", "60", "80", "inf"), default="60")
    timing.add_argument("--workers", type=int, choices=(1,), default=1)
    timing.add_argument(
        "--confirm-isolated", action="store_true",
        help="Operator declaration: competing experiments are stopped. The runner cannot verify system-wide isolation.",
    )
    timing.set_defaults(run=run_timing_setup)
    args = parser.parse_args(argv)
    if args.trial_start < 0:
        parser.error("--trial-start must be nonnegative")
    args.run(args)


if __name__ == "__main__":
    main()
