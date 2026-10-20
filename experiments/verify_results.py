"""Read-only checks of complete recorded trials, provenance, and manuscript values.

Run from the repository root with ``python -m experiments.verify_results``.
No experiments are rerun, and no files are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import warnings

import numpy as np


EXPECTED_ROWS = {
    "baseline_tuning": 2640, "gaussian": 24000, "poisson": 7000,
    "mixed": 7000, "separation": 2500, "background": 1000,
    "timing": 1350, "timing_setup": 90, "validation": 1800,
    "case2d": 2400, "timing_80db": 60, "conditioning": 4000,
    "blank_calibration": 3000, "model_stress": 26500,
    "beam_ablation": 4000, "unknown_sparsity": 1000,
    "stability_check": 3000, "tuning_reproduction": 2640,
}
METHODS = ("joint", "full_linear", "pie", "htwf", "htaf")
CONDITIONS = ("suite", "sparsity", "N", "snr", "photons", "stress", "value",
              "method", "config_id", "total_photons_two_frames", "beam_width",
              "width_factor", "calibration", "calibration_total_photons")
STATISTICS = {"min", "max", "mean", "median", "q10_q90", "q25_q75"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
PRIVATE_TEXT = re.compile(
    r"(?:/Users/|/home/|[A-Za-z]:[\\/]Users[\\/])"
    r"|(?:[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)"
    r"|(?:\b(?:19|20)\d{2}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?)"
)
PRIVATE_KEYS = re.compile(r"^(?:created|updated|started|finished|completed|exported)_(?:utc|at|timestamp|date)$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same(left, right):
    if left == right or left in (None, "") and right in (None, ""):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def select(rows, **conditions):
    return [row for row in rows if all(same(row.get(key), value) for key, value in conditions.items())]


def numeric_values(rows, field):
    return np.asarray([float(row[field]) for row in rows if row.get(field) not in (None, "")], dtype=float)


def support_rate(rows):
    require(bool(rows), "Empty support condition")
    values = [row["support_exact"] for row in rows if row.get("support_exact") not in (None, "")]
    require(all(value in ("0", "1", 0, 1) for value in values), "Invalid support outcome")
    return sum(int(value) for value in values) / len(values) if values else None


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def close(actual, expected, tolerance, message):
    require(actual is not None and expected is not None and abs(actual - expected) <= tolerance,
            f"{message}: {actual} != {expected}")


def equivalent(actual, expected, message):
    """JSON null represents a recorded undefined/nonfinite aggregate, never zero."""
    if isinstance(actual, (list, tuple, np.ndarray)):
        require(isinstance(expected, (list, tuple)) and len(actual) == len(expected), message)
        for index, (left, right) in enumerate(zip(actual, expected)):
            equivalent(left, right, f"{message}[{index}]")
        return
    if expected is None:
        require(actual is None or isinstance(actual, (float, np.floating)) and not math.isfinite(actual), message)
        return
    require(actual is not None and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14),
            f"{message}: {actual} != {expected}")


def statistic(values, key):
    if not len(values):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if key in ("q10_q90", "q25_q75"):
            return np.quantile(values, [.1, .9] if key == "q10_q90" else [.25, .75]).tolist()
        return float(getattr(np, key)(values))


def verify_distribution(values, record, message):
    """Check both historical distribution formats without dropping failure rows."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if "nonfinite_count" in record:
        require(record["count"] == len(values), f"{message}: total count")
        require(record["nonfinite_count"] == len(values) - len(finite), f"{message}: nonfinite count")
    elif "total_rows" in record:
        require(record["total_rows"] == len(values), f"{message}: total rows")
        require(record["finite_count"] == len(finite), f"{message}: finite count")
        require(record["unavailable_or_infinite_count"] == len(values) - len(finite), f"{message}: unavailable count")
    elif "count" in record:
        require(record["count"] == len(finite), f"{message}: finite count")
    for key in STATISTICS & record.keys():
        equivalent(statistic(finite, key), record[key], f"{message}/{key}")


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center-half), min(1., center+half)]


def _verify_stability_group(part, group, suite):
    checked = [row for row in part if row["check_status"] == "checked"]
    sufficient = [row for row in checked if row["amplitude_condition"] == "1"]
    triangle = [row for row in checked if row["triangle_amplitude_condition"] == "1"]
    counts = {
        "checked": len(checked), "unavailable": len(part)-len(checked),
        "original_rank_drop_rows": sum(bool(float(row["rank_dropped_support_count"] or 0)) for row in part),
        "bound_violations": sum(row["theorem_bound_holds"] == "0" for row in checked),
        "triangle_bound_violations": sum(row["triangle_bound_holds"] == "0" for row in checked),
        "actual_exact_supports": sum(row["support_exact"] == "1" for row in part),
        "sufficient_amplitude_count": len(sufficient),
        "sufficient_and_exact_count": sum(row["support_exact"] == "1" for row in sufficient),
        "triangle_sufficient_amplitude_count": len(triangle),
        "triangle_sufficient_and_exact_count": sum(row["support_exact"] == "1" for row in triangle),
    }
    for key, actual in counts.items():
        require(group[key] == actual, f"{suite}/{key}")


def verify_summary(rows, summary, suite):
    require(summary["rows"] == len(rows), f"{suite}: summary row count")
    coverage = np.zeros(len(rows), dtype=int)
    aliases = {"nrmse_all_returned_estimates": "perturbation_nrmse",
               "nrmse_when_algorithm_passed": "perturbation_nrmse",
               "identity_error": "raw_clean_lag_identity_relative_error"}
    for group in summary["aggregate"]:
        conditions = {key: group[key] for key in CONDITIONS if key in group}
        indices = [index for index, row in enumerate(rows)
                   if all(same(row.get(key), value) for key, value in conditions.items())]
        part = [rows[index] for index in indices]
        require(len(part) == group["trials"], f"{suite}: condition count {conditions}")
        coverage[indices] += 1
        support = [row["support_exact"] for row in part if row.get("support_exact") not in (None, "")]
        successes = sum(int(value) for value in support)
        for rate_key in ("support_success_rate", "exact_support_rate"):
            if rate_key in group:
                equivalent(support_rate(part), group[rate_key], f"{suite}: {rate_key}")
        for key in ("support_wilson95", "wilson_95_interval"):
            if key in group:
                equivalent(wilson(successes, len(support)), group[key], f"{suite}: {key}")
        for key, actual in (("support_applicable_trials", len(support)), ("successes", successes),
                            ("failures", sum(int(row.get("failed") or 0) for row in part))):
            if key in group:
                require(group[key] == actual, f"{suite}: {key}")
        if "algorithm_pass_rate" in group:
            equivalent(sum(int(row["algorithm_passed"]) for row in part) / len(part), group["algorithm_pass_rate"], f"{suite}: algorithm pass rate")
        for key, field in (("status_counts", "check_status" if suite == "stability_check" else "status"),
                           ("original_status_counts", "original_status")):
            if key in group:
                require(dict(Counter(row[field] for row in part)) == group[key], f"{suite}: {key}")
        for key, expected in group.items():
            if isinstance(expected, dict) and (key in part[0] or key in aliases):
                selected = [row for row in part if row["algorithm_passed"] == "1"] if key == "nrmse_when_algorithm_passed" else part
                field = aliases.get(key, key)
                values = [float(row[field]) if row.get(field) not in (None, "") else float("nan") for row in selected]
                verify_distribution(values, expected, f"{suite}/{key}")
            for suffix in ("_median", "_mean", "_q10_q90", "_q25_q75", "_available"):
                if key.endswith(suffix) and key[:-len(suffix)] in part[0]:
                    field = key[:-len(suffix)]
                    values = numeric_values(part, field)
                    if suffix == "_available":
                        require(len(values) == expected, f"{suite}: {key}")
                    else:
                        equivalent(statistic(values, suffix[1:]), expected, f"{suite}: {key}")
        if suite == "stability_check":
            _verify_stability_group(part, group, suite)
    require(np.all(coverage == 1), f"{suite}: each row must belong to exactly one summary condition")
    if "method_trials" in summary:
        require(summary["method_trials"] == len(rows), f"{suite}: method-trial count")
    if "jobs_completed" in summary:
        require(summary["jobs_completed"] == len({row["id"] for row in rows}), f"{suite}: completed-job count")
    if "failures" in summary:
        require(summary["failures"] == sum(int(row.get("failed") or 0) for row in rows), f"{suite}: total failures")


def verify_public_metadata(value, name):
    if isinstance(value, dict):
        for key, child in value.items():
            require(not PRIVATE_KEYS.fullmatch(key), f"{name}: private calendar field {key}")
            require(not PRIVATE_TEXT.search(key), f"{name}: private metadata identifier")
            verify_public_metadata(child, name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            verify_public_metadata(child, name)
    elif isinstance(value, str):
        require(not PRIVATE_TEXT.search(value), f"{name}: private path, email, or calendar timestamp")


def verify_hash_map(hashes, label):
    require(isinstance(hashes, dict) and bool(hashes), f"{label}: missing provenance")
    for name, value in hashes.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, f"{label}: invalid file identifier")
        require(isinstance(value, str) and HEX_SHA256.fullmatch(value), f"{label}: invalid SHA-256")


def verify_trial_identities(rows, suite, manifest):
    if "id" in rows[0]:
        fields = ("id", "method")
    elif suite == "validation":
        fields = ("N", "trial", "method")
    elif suite == "case2d":
        fields = ("total_photons_two_frames", "trial", "method")
    else:
        fields = tuple(key for key in CONDITIONS if key in rows[0]) + ("trial",)
    require(len({tuple(row[key] for key in fields) for row in rows}) == len(rows), f"{suite}: duplicate trial identities")
    if "jobs" in manifest:
        require({row["id"] for row in rows} == {job["id"] for job in manifest["jobs"]}, f"{suite}: job coverage")


def verify_timing_quantiles(directory, rows):
    quantiles = json_file(directory / "paired_quantiles.json")
    require(quantiles["trials_per_method"] == 30, "80-dB timing count")
    for method in ("joint", "full_linear"):
        part = select(rows, method=method)
        verify_distribution(1000 * numeric_values(part, "online_seconds"), quantiles["online_ms"][method], f"80-dB {method} online")
        require(sum(int(row["support_exact"]) for row in part) == quantiles["support_successes"][method], "80-dB timing successes")
    full = select(rows, method="full_linear")
    verify_distribution(1000 * numeric_values(full, "setup_seconds"), quantiles["full_linear_setup_ms"], "80-dB full-linear setup")
    verify_distribution(1000 * numeric_values(full, "algorithm_total_seconds"), quantiles["full_linear_setup_plus_online_ms"], "80-dB full-linear total")
    paired = {method: {row["trial"]: float(row["online_seconds"]) for row in select(rows, method=method)} for method in ("joint", "full_linear")}
    require(paired["joint"].keys() == paired["full_linear"].keys(), "80-dB timing pairing")
    ratios = [paired["full_linear"][trial] / value for trial, value in paired["joint"].items()]
    verify_distribution(ratios, quantiles["paired_online_speedup"], "80-dB paired online ratios")


def verify_arrays(root, data):
    with np.load(root / "results/validation/raw_data.npz", allow_pickle=False) as arrays:
        fields = ("clean_intensities", "quarter_intensities", "supports", "amplitudes")
        require(set(arrays.files) == {f"N{n}_{field}" for n in (1, 2, 3) for field in fields}, "Validation array identities")
        for n in (1, 2, 3):
            require(arrays[f"N{n}_clean_intensities"].shape == (200, 2, 257), "Validation frame shape")
            require(arrays[f"N{n}_quarter_intensities"].shape == (200, 257), "Validation control frame shape")
            require(arrays[f"N{n}_supports"].shape == arrays[f"N{n}_amplitudes"].shape == (200, n), "Validation truth shape")
            for row in select(data["validation"], N=n):
                trial = int(row["trial"])
                require(json.loads(row["true_support"]) == arrays[f"N{n}_supports"][trial].tolist(), "Validation support/CSV pairing")
                amplitudes = arrays[f"N{n}_amplitudes"][trial]
                require(np.array_equal(json.loads(row["amplitudes_real"]), amplitudes.real), "Validation real amplitude/CSV pairing")
                require(np.array_equal(json.loads(row["amplitudes_imag"]), amplitudes.imag), "Validation imaginary amplitude/CSV pairing")
    with np.load(root / "results/case2d/measured_intensities.npz", allow_pickle=False) as arrays:
        require(arrays["intensities"].shape == (1200, 2, 65, 65), "Two-dimensional frame shape")
        identities = {(float(total), int(trial), int(seed)) for total, trial, seed in zip(arrays["total_photons_two_frames"], arrays["trial"], arrays["seeds"])}
        require(len(identities) == 1200, "Two-dimensional input identities")
        require(identities == {(float(row["total_photons_two_frames"]), int(row["trial"]), int(row["seed"])) for row in data["case2d"]}, "Two-dimensional frame/CSV pairing")
    conditioning = root / "results/conditioning"
    summary = json_file(conditioning / "summary.json")["exact_N2_beta"]
    with np.load(conditioning / "beta_N2_all_support_unions.npz", allow_pickle=False) as arrays:
        count = summary["distinct_nonempty_unions"]
        require(arrays["matrix"].shape == (16, 42), "Restricted-gain operator shape")
        require(arrays["gains"].shape == arrays["conditions"].shape == arrays["union_sizes"].shape == (count,), "Restricted-gain union count")
        require(arrays["union_positions"].shape == (count, 4), "Restricted-gain union positions")
        minimum_index = int(np.argmin(arrays["gains"]))
        equivalent(float(arrays["gains"][minimum_index]), summary["beta_numerical_complete_enumeration"], "Restricted gain")
        require(arrays["union_positions"][minimum_index, :int(arrays["union_sizes"][minimum_index])].tolist() == summary["minimizing_union"], "Minimum-gain union identity")
        equivalent(float(arrays["conditions"][minimum_index]), summary["minimizing_union_condition"], "Minimum-gain union condition")
    with np.load(conditioning / "raw_dictionaries_and_singular_values.npz", allow_pickle=False) as arrays:
        for row in data["conditioning"]:
            prefix = f"N{int(row['N'])}_factor{float(row['width_factor']):g}"
            singular = arrays[f"{prefix}_singular_values"][int(row["trial"])]
            singular = singular[np.isfinite(singular)]
            require(len(singular) == 2 * int(row["union_size"]), "Conditioning union dimension")
            equivalent(float(singular[-1]), float(row["minimum_gain"]), "Conditioning gain/CSV pairing")
            equivalent(float(singular[0]), float(row["maximum_gain"]), "Conditioning maximum/CSV pairing")
            equivalent(float(singular[0] / singular[-1]), float(row["condition"]), "Conditioning ratio/CSV pairing")


def verify_stability_rows(data):
    """Check saved bounds and their flags, using the recorded estimator outputs."""
    fields = ("trial", "snr", "calibration", "calibration_total_photons")
    blanks = {tuple(row[key] for key in fields): row for row in data["blank_calibration"]}
    require(len(blanks) == len(data["stability_check"]), "Stability/blank trial coverage")
    for row in data["stability_check"]:
        original = blanks[tuple(row[key] for key in fields)]
        for key in ("support_exact", "estimated_support", "rank_dropped_support_count"):
            require(row[key] == original[key], f"Stability original {key}")
        require(row["original_status"] == original["status"] and row["original_failed"] == original["failed"], "Stability original status")
        if row["check_status"] != "checked":
            continue
        epsilon, residual, lhs, beta = (float(row[key]) for key in (
            "actual_raw_error_norm", "reported_raw_residual", "coefficient_error_from_original_nrmse", "beta_numerical"))
        equivalent(residual, float(original["raw_lag_residual"]), "Stability original residual")
        rho = max(0., residual - epsilon)
        bound = (2 * epsilon + rho) / beta
        triangle = (epsilon + residual) / beta
        for key, actual in (("rho", rho), ("theorem_bound", bound), ("triangle_bound", triangle)):
            equivalent(actual, float(row[key]), f"Stability {key}")
        amplitude = float(row["minimum_true_amplitude"])
        for key, actual in (("theorem_bound_holds", lhs <= bound), ("triangle_bound_holds", lhs <= triangle),
                            ("amplitude_condition", amplitude > bound), ("triangle_amplitude_condition", amplitude > triangle)):
            require(int(row[key]) == int(actual), f"Stability {key}")


def verify_files(root):
    root = Path(root)
    data, authenticated_sources = {}, {}
    for suite, count in EXPECTED_ROWS.items():
        directory = root / "results" / suite
        manifest = json_file(directory / "manifest.json")
        require(manifest["suite"] == suite and manifest["rows"] == count, f"{suite}: manifest counts")
        verify_public_metadata(manifest, f"{suite}/manifest.json")
        hashes = manifest["data_sha256"]
        verify_hash_map(hashes, f"{suite}/data_sha256")
        expected_files = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() and path.name not in ("manifest.json", "frozen.json")}
        require(set(hashes) == expected_files, f"{suite}: incomplete data-file hash coverage")
        for name, expected in hashes.items():
            require(sha256(directory / name) == expected, f"{suite}/{name}: data hash")
            if Path(name).suffix in (".json", ".csv"):
                verify_public_metadata(json_file(directory / name) if Path(name).suffix == ".json" else (directory / name).read_text(), f"{suite}/{name}")
        provenance = manifest.get("recorded_sources", manifest.get("recorded_model_sources"))
        verify_hash_map(provenance, f"{suite}/recorded_sources")
        if "recorded_source_file_sha256" in manifest:
            verify_hash_map(manifest["recorded_source_file_sha256"], f"{suite}/recorded_source_file_sha256")
        distribution = manifest["distribution_source_hashes"]
        verify_hash_map(distribution, f"{suite}/distribution_source_hashes")
        for name, expected in distribution.items():
            if name in authenticated_sources:
                require(authenticated_sources[name] == expected, "Inconsistent distribution source hashes")
            else:
                require(sha256(root / name) == expected, f"{name}: distributed source hash")
                authenticated_sources[name] = expected
        with (directory / "trials.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == count, f"{suite}: CSV row count")
        verify_trial_identities(rows, suite, manifest)
        verify_summary(rows, json_file(directory / "summary.json"), suite)
        data[suite] = rows
    frozen_path = root / "results/baseline_tuning/frozen.json"
    tuning_manifest = json_file(frozen_path.parent / "manifest.json")
    require(sha256(frozen_path) == tuning_manifest["frozen_sha256"], "Frozen settings hash")
    frozen = json_file(frozen_path)
    verify_public_metadata(frozen, "baseline_tuning/frozen.json")
    require(frozen["distribution_source_hashes"] == authenticated_sources, "Frozen distribution source identity")
    verify_hash_map(frozen["recorded_solver_sources"], "Frozen recorded solver sources")
    for method in ("pie", "htwf", "htaf"):
        settings = frozen["baseline_configs"][method]
        require(settings["iterations"] == 1200 and settings["restarts"] == 32 and settings["threshold_after"] == 200, "Iterative baseline budget")
    require(frozen["baseline_configs"]["full_linear"] == {"iterations": 12, "restarts": 1}, "Full-linear budget")
    verify_arrays(root, data)
    verify_timing_quantiles(root / "results/timing_80db", data["timing_80db"])
    verify_stability_rows(data)
    return data


def verify_paper(data):
    validation = data["validation"]
    for n in (1, 2, 3):
        close(support_rate(select(validation, N=n, method="joint_global")), 1, 0, "Noiseless joint support")
        close(support_rate(select(validation, N=n, method="spectral_algebraic")), 1 if n < 3 else 29 / 200, 0, "Spectral support")
    gaussian = data["gaussian"]
    for method, expected in (("joint", 1.163628036346777e-4), ("full_linear", 2.072729e-4)):
        accuracy = select(gaussian, sparsity=2, snr=80, method=method)
        require(len(accuracy) == 200, "80-dB accuracy count")
        close(support_rate(accuracy), 1, 0, "80-dB exact support")
        close(median(accuracy, "defect_nrmse"), expected, 5e-11, "80-dB NRMSE")
    timing80 = data["timing_80db"]
    joint_ms = 1000 * median(select(timing80, method="joint"), "online_seconds")
    linear_ms = 1000 * median(select(timing80, method="full_linear"), "online_seconds")
    close(joint_ms, .1971459132619202, 1e-12, "80-dB serial joint online time")
    close(linear_ms, 15.258395462296903, 1e-12, "80-dB serial full-linear online time")
    require(77 < linear_ms / joint_ms < 78, "80-dB online median ratio")
    for method, rates in (("joint", (.995, .74, .545)), ("full_linear", (1, .995, .95))):
        for n, rate in enumerate(rates, 1):
            close(support_rate(select(gaussian, sparsity=n, snr=40, method=method)), rate, 0, "40-dB support")
    table_rates = (.94, 1, .49, .27, .395)
    table_nrmse = (.00116, .00190, .488, .713, .639)
    table_times_ms = (.209, 14.3, 1101, 3206, 2389)
    table_tolerances = (.0005, .05, .5, .5, .5)
    for index, method in enumerate(METHODS):
        accuracy = select(gaussian, sparsity=2, snr=60, method=method)
        timed = select(data["timing"], sparsity=2, snr=60, method=method)
        close(support_rate(accuracy), table_rates[index], 0, "Table support")
        close(median(accuracy, "defect_nrmse"), table_nrmse[index], 5e-6 if index < 2 else .0005, "Table NRMSE")
        close(1000 * median(timed, "online_seconds"), table_times_ms[index], table_tolerances[index], "Table time")
    for method, mean, tolerance in (("joint", .0863, .00005), ("full_linear", .00510, .000005)):
        part = select(gaussian, sparsity=2, snr=60, method=method)
        close(statistics.mean(float(row["defect_nrmse"]) for row in part), mean, tolerance, "Mean NRMSE")
    for distance, rate in ((2, 0), (4, 0), (5, 1), (6, 1), (8, 1)):
        close(support_rate(select(data["separation"], value=distance, method="joint")), rate, 0, "Separation support")
    for method, rates in (("joint", (.96, .51)), ("full_linear", (1, .92))):
        for value, rate in zip((0, .001), rates):
            close(support_rate(select(data["background"], value=value, method=method)), rate, 0, "Background support")
    for method, rate in (("calibrated", .915), ("gamma_plus_2pct", .78)):
        close(support_rate(select(data["case2d"], total_photons_two_frames=2e9, method=method)), rate, 0, "Two-dimensional support")
    for n, expected in ((1, .33), (2, .95), (3, 5.1)):
        actual = 1000 * median(select(data["timing_setup"], sparsity=n), "total_cold_reconstruction_seconds")
        close(actual, expected, .005 if n < 3 else .05, "Fresh setup plus reconstruction")
    times = [1000 * median(select(data["timing"], sparsity=n, snr=snr, method="joint"), "online_seconds") for n in (1, 2, 3) for snr in (20, 60, "inf")]
    close(min(times), .153, .0005, "Minimum joint online median")
    close(max(times), .383, .0005, "Maximum joint online median")
    for suite in ("poisson", "mixed"):
        for photons in sorted({row["photons"] for row in data[suite]}):
            require(support_rate(select(data[suite], photons=photons, method="full_linear")) > support_rate(select(data[suite], photons=photons, method="joint")), f"{suite}: photon reliability ordering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        data = verify_files(args.root.resolve())
        verify_paper(data)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(f"PASS: {sum(map(len, data.values())):,} per-trial rows across {len(data)} studies.")
    print("Data/source hashes, all condition summaries, array pairing, and manuscript values agree.")
    print("Private paths, email addresses, and calendar timestamps are absent from result metadata.")
    print("Recorded provenance is separate from distributed source hashes; timings are not rerun.")


if __name__ == "__main__":
    main()
