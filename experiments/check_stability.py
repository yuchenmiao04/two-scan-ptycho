"""Postprocess saved blank-calibration trials against the unknown-support bound.

No baseline or reconstruction is rerun. The actual error is b - A v(truth),
including FFT/model roundoff, not b minus an independently computed clean b.
The finite-family gain is a floating-point SVD value, NOT an interval certificate.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from twoscan_ptycho.core import position_to_index
from twoscan_ptycho.reconstruction import build_joint_dictionary, GlobalSupportSolver


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(values):
    values = list(values)
    finite = [float(x) for x in values if x is not None and np.isfinite(x)]
    result = {"total_rows": len(values), "finite_count": len(finite),
              "unavailable_or_infinite_count": len(values)-len(finite)}
    if finite:
        result.update(min=float(np.min(finite)), median=float(np.median(finite)),
                      max=float(np.max(finite)), q10_q90=np.quantile(finite, [.1, .9]).tolist())
    return result


def support(text):
    return tuple(int(item) for item in text.split(";") if item)


def raw_from_cw(measured, cw, lag_count):
    block = (np.fft.ifft(measured, axis=1) - cw)[:, 1:lag_count+1]
    return np.concatenate([part for frame in block for part in (frame.real, frame.imag)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/reproduced/blank_calibration")
    parser.add_argument("--conditioning", type=Path, default=ROOT / "results/conditioning")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/stability_check")
    args = parser.parse_args()
    if not (args.input / "measurements").is_dir() or not (args.input / "calibrations").is_dir():
        parser.error("Raw measurements are required: run experiments.run_calibration blank --output results/reproduced/blank_calibration, then pass that directory with --input.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing postprocessing: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    blank_manifest = read_json(args.input / "manifest.json")
    blank_summary = read_json(args.input / "summary.json")
    condition_summary = read_json(args.conditioning / "summary.json")
    settings = blank_manifest["settings"]
    exact = condition_summary["exact_N2_beta"]
    required = {"n": 257, "sparsity": 2, "candidate_bound": 10, "gamma": .05}
    for key, value in required.items():
        if settings[key] != value:
            raise ValueError(f"Blank operator setting mismatch: {key}")
    if settings["positions"] != [0, 1] or exact["scans"] != [0, 1]:
        raise ValueError("The two datasets do not use the same scan centers")
    if (exact["n"], exact["N"], exact["K"], exact["gamma"]) != (257, 2, 21, .05):
        raise ValueError("Gain was evaluated for a different operator/class")
    with (args.input / "trials.csv").open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    if len(original) != blank_summary["rows"]:
        raise ValueError("Original summary/CSV row count mismatch")
    expected_conditions = {(str(snr), kind, "" if budget is None else str(float(budget)))
                           for snr in settings["object_effective_snr_db"]
                           for kind, budget in [("exact", None)] + [("poisson", x) for x in settings["calibration_total_expected_photons"]]}
    observed = Counter((row["snr"], row["calibration"], row["calibration_total_photons"]) for row in original)
    if set(observed) != expected_conditions or any(count != settings["trials_per_condition"] for count in observed.values()):
        raise ValueError("The saved blank-calibration suite is incomplete")
    identities = [(row["trial"], row["snr"], row["calibration"], row["calibration_total_photons"]) for row in original]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicated original trial/condition rows")

    model = build_joint_dictionary(257, .05, (0, 1), range(-10, 11), 2,
                                   minimum_separation=4, enforce_margin=True)
    # Building once checks class membership/counts; no call to solve is made.
    legal_solver = GlobalSupportSolver(model, 2)
    legal_supports = {tuple(sorted(s)) for s in legal_solver.all_supports}
    gain_path = args.conditioning / "beta_N2_all_support_unions.npz"
    with np.load(gain_path) as archive:
        matrix = archive["matrix"].copy()
        positions = archive["candidate_positions"].copy()
        gains = archive["gains"].copy()
        union_count = len(archive["union_sizes"])
    if tuple(positions) != model.candidates or matrix.shape != model.matrix.shape:
        raise ValueError("Archived gain matrix ordering/shape differs from the blank model")
    discrepancy = float(np.max(np.abs(matrix-model.matrix)))
    if not np.allclose(matrix, model.matrix, rtol=64*np.finfo(float).eps, atol=1e-300):
        raise ValueError("Archived gain matrix values differ from the blank model")
    if union_count != 5894 or legal_solver.candidate_count != 158:
        raise ValueError("Support class enumeration does not match the gain archive")
    beta = float(np.min(gains))
    if beta <= 0 or not np.isclose(beta, exact["beta_numerical_complete_enumeration"], rtol=1e-14, atol=0):
        raise ValueError("Gain summary/archive mismatch or nonpositive numerical gain")
    for name in ("src/twoscan_ptycho/reconstruction.py", "src/twoscan_ptycho/core.py"):
        if sha256(ROOT / name) != blank_manifest.get("distribution_source_hashes", blank_manifest.get("source_hashes", {}))[name]:
            raise ValueError(f"Recovery/model source changed since the blank run: {name}")
    exact_cw = np.fft.ifft(model.background_intensities, axis=1)
    def input_id(path):
        path = Path(path)
        for label, base in (("blank_calibration", args.input), ("conditioning", args.conditioning)):
            try:
                return str(Path(label) / path.relative_to(base))
            except ValueError:
                pass
        raise ValueError("Input is outside the declared dataset directories")
    inputs = {input_id(path): sha256(path) for path in
              (args.input / "manifest.json", args.input / "summary.json", args.input / "trials.csv",
               args.conditioning / "manifest.json", args.conditioning / "summary.json", gain_path)}
    measured_cache, blank_cache = {}, {}
    rows = []
    for old in original:
        trial, snr = int(old["trial"]), old["snr"]
        measurement_path = args.input / "measurements" / f"trial{trial:05d}_snr{snr}.npz"
        if measurement_path not in measured_cache:
            inputs[input_id(measurement_path)] = sha256(measurement_path)
            with np.load(measurement_path) as data:
                measured_cache[measurement_path] = {key: data[key].copy() for key in data.files}
        saved = measured_cache[measurement_path]
        truth, measured = saved["truth"], saved["measured"]
        reasons = []
        if array_hash(truth) != old["truth_sha256"] or array_hash(measured) != old["measurement_sha256"]:
            reasons.append("original_array_hash_mismatch")
        true_support = tuple(sorted(support(old["true_support"])))
        estimated_support = tuple(sorted(support(old.get("estimated_support", ""))))
        truth_grid_support = tuple(int(x) for x in model.candidates if truth[position_to_index(x, 257)] != 0)
        if true_support != truth_grid_support or len(true_support) != 2 or true_support not in legal_supports:
            reasons.append("true_support_outside_declared_class")
        if estimated_support not in legal_supports:
            reasons.append("estimated_support_outside_declared_class")
        if old["calibration"] == "exact":
            cw = exact_cw
        else:
            budget = float(old["calibration_total_photons"])
            blank_path = args.input / "calibrations" / f"trial{trial:05d}_photons{budget:g}.npz"
            if blank_path not in blank_cache:
                inputs[input_id(blank_path)] = sha256(blank_path)
                with np.load(blank_path) as data:
                    blank_cache[blank_path] = (data["estimated_cw"].copy(), int(data["seed"]))
            cw, blank_seed = blank_cache[blank_path]
            if blank_seed != int(old["blank_seed"]):
                reasons.append("blank_seed_mismatch")
        b = raw_from_cw(measured, cw, model.lag_count)
        coefficients = truth[position_to_index(positions, 257)]
        physical_vector = np.r_[coefficients.real, coefficients.imag]
        epsilon = float(np.linalg.norm(b - matrix @ physical_vector))
        model_roundoff = float(np.linalg.norm(raw_from_cw(saved["noiseless"], exact_cw, model.lag_count) - matrix @ physical_vector))
        original_status = old.get("status", "")
        if not original_status:
            reasons.append("missing_original_solver_status")
        if old.get("failed") != "0":
            reasons.append("original_numerical_exception")
        rank_drops = int(old["rank_dropped_support_count"]) if old.get("rank_dropped_support_count") else None
        if rank_drops is None:
            reasons.append("missing_original_rank_drop_count")
        try:
            residual = float(old["raw_lag_residual"])
            lhs = float(old["defect_nrmse"]) * float(np.linalg.norm(truth))
            amplitude_min = float(np.min(np.abs(coefficients[np.abs(coefficients) > 0])))
            if not all(np.isfinite(x) and x >= 0 for x in (residual, lhs, epsilon, amplitude_min)):
                raise ValueError("nonfinite metric")
            rho = max(0., residual - epsilon)
            bound = (2*epsilon + rho) / beta
            tight = (epsilon + residual) / beta
            support_exact = int(old["support_exact"])
            if support_exact != int(estimated_support == true_support):
                reasons.append("original_support_metric_mismatch")
        except (KeyError, ValueError):
            reasons.append("unavailable_original_reconstruction_metrics")
            residual = lhs = amplitude_min = rho = bound = tight = support_exact = None
        row = {"trial": trial, "snr": snr, "calibration": old["calibration"],
               "calibration_total_photons": old["calibration_total_photons"],
               "original_status": original_status, "original_failed": int(old["failed"]),
               "rank_dropped_support_count": rank_drops, "check_status": "checked" if not reasons else ";".join(reasons),
               "estimated_support": old.get("estimated_support", ""), "support_exact": support_exact,
               "actual_raw_error_norm": epsilon, "noiseless_identity_roundoff_norm": model_roundoff,
               "previous_b_minus_clean_b_norm": float(old["combined_raw_lag_error_norm"]),
               "reported_raw_residual": residual, "coefficient_error_from_original_nrmse": lhs,
               "beta_numerical": beta, "rho": rho, "theorem_bound": bound, "triangle_bound": tight,
               "minimum_true_amplitude": amplitude_min,
               "theorem_bound_holds": int(lhs <= bound) if not reasons else None,
               "triangle_bound_holds": int(lhs <= tight) if not reasons else None,
               "amplitude_condition": int(amplitude_min > bound) if not reasons else None,
               "triangle_amplitude_condition": int(amplitude_min > tight) if not reasons else None,
               "theorem_bound_over_error": bound/lhs if not reasons and lhs > 0 else None,
               "triangle_bound_over_error": tight/lhs if not reasons and lhs > 0 else None,
               "zero_error": int(lhs == 0) if lhs is not None else None}
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["snr"], row["calibration"], row["calibration_total_photons"])].append(row)
    aggregate = []
    for key, part in groups.items():
        checked = [row for row in part if row["check_status"] == "checked"]
        sufficient = [row for row in checked if row["amplitude_condition"]]
        tight_sufficient = [row for row in checked if row["triangle_amplitude_condition"]]
        item = {"snr": key[0], "calibration": key[1], "calibration_total_photons": key[2],
                "trials": len(part), "checked": len(checked), "unavailable": len(part)-len(checked),
                "status_counts": dict(Counter(row["check_status"] for row in part)),
                "original_status_counts": dict(Counter(row["original_status"] for row in part)),
                "original_rank_drop_rows": sum(bool(row["rank_dropped_support_count"]) for row in part),
                "bound_violations": sum(not row["theorem_bound_holds"] for row in checked),
                "triangle_bound_violations": sum(not row["triangle_bound_holds"] for row in checked),
                "actual_exact_supports": sum(row["support_exact"] == 1 for row in part),
                "sufficient_amplitude_count": len(sufficient),
                "sufficient_and_exact_count": sum(row["support_exact"] for row in sufficient),
                "triangle_sufficient_amplitude_count": len(tight_sufficient),
                "triangle_sufficient_and_exact_count": sum(row["support_exact"] for row in tight_sufficient)}
        for name in ("actual_raw_error_norm", "coefficient_error_from_original_nrmse", "theorem_bound", "triangle_bound",
                     "theorem_bound_over_error", "triangle_bound_over_error", "noiseless_identity_roundoff_norm"):
            item[name] = summarize(row[name] for row in part)
        aggregate.append(item)
        print(f"SNR={key[0]}, blank={key[2] or 'exact'}: checked={len(checked)}, violations={item['bound_violations']}, sufficient={len(sufficient)}, exact={item['actual_exact_supports']}", flush=True)

    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    statement = ("A floating-point consistency check using actual combined raw error, the saved original estimator's residual and NRMSE, "
                 "and a numerically enumerated restricted gain. It is not an interval-arithmetic certificate, "
                 "not a new proof of the theorem, and not a deployable certificate because epsilon uses simulated ground truth.")
    summary = {"rows": len(rows), "aggregate": aggregate, "interpretation": statement,
               "operator": {"n": 257, "gamma": .05, "scan_centers": [0, 1], "candidate_positions": positions.tolist(),
                            "upper_sparsity": 2, "strict_separation": ">4", "positive_lags_per_frame": 4,
                            "row_order": "scan0 real lags, scan0 imaginary lags, scan1 real lags, scan1 imaginary lags",
                            "coefficient_norm": "physical complex Euclidean norm = stacked real/imag Euclidean norm",
                            "support_count_including_empty": 158, "nonempty_union_count": union_count,
                            "beta_numerical": beta, "current_vs_archived_matrix_max_difference": discrepancy},
               "formulas": {"epsilon": "norm(b - A v(truth))", "rho": "max(0, reported_raw_residual - epsilon)",
                            "theorem_bound": "(2 epsilon + rho)/beta_num", "triangle_bound": "(epsilon + reported_raw_residual)/beta_num",
                            "lhs": "saved_defect_nrmse * norm(saved_truth)"},
               "estimator_policy": "Original estimates/results are not changed. No reconstruction solver or baseline was rerun.",
               "all_rows_accounted_for": len(rows) == len(original),
               "rank_drops_do_not_imply_row_deletion": True}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "manifest.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "workers": 1,
               "script_sha256": sha256(__file__), "input_sha256": inputs,
               "model_source_sha256": {name: sha256(ROOT / name) for name in
                                       ("src/twoscan_ptycho/reconstruction.py", "src/twoscan_ptycho/core.py")},
               "scope": statement})
    checked_rows = [row for row in rows if row["check_status"] == "checked"]
    violations = [row for row in checked_rows if not row["theorem_bound_holds"] or not row["triangle_bound_holds"] or
                  row["amplitude_condition"] and not row["support_exact"]]
    if violations or len(checked_rows) != len(rows):
        raise SystemExit("Postprocessing retained numerical violations or unavailable rows; inspect all output rows.")


if __name__ == "__main__":
    main()
