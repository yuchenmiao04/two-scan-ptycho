"""Read-only verification of the recorded numerical data and manuscript values.

Run from the repository root with ``python -m experiments.verify_results``.
No experiments are rerun, and no files are changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


EXPECTED_ROWS = {
    "baseline_tuning": 2640, "gaussian": 24000, "poisson": 7000,
    "mixed": 7000, "separation": 2500, "background": 1000,
    "timing": 1350, "timing_setup": 90, "validation": 1200,
    "case2d": 2400,
}
METHODS = ("joint", "full_linear", "pie", "htwf", "htaf")
CONDITIONS = ("suite", "sparsity", "N", "snr", "photons", "stress", "value",
              "method", "config_id", "total_photons_two_frames")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same(left, right):
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def select(rows, **conditions):
    return [row for row in rows if all(same(row.get(key), value) for key, value in conditions.items())]


def support_rate(rows):
    require(bool(rows), "Empty support condition")
    require(all(row["support_exact"] in ("0", "1") for row in rows), "Unavailable support outcome")
    return sum(int(row["support_exact"]) for row in rows) / len(rows)


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def close(actual, expected, tolerance, message):
    require(abs(actual - expected) <= tolerance, f"{message}: {actual} != {expected}")


def verify_summary(rows, summary, suite):
    require(summary["rows"] == len(rows), f"{suite}: summary row count")
    covered = 0
    for group in summary["aggregate"]:
        part = select(rows, **{key: group[key] for key in CONDITIONS if key in group})
        require(len(part) == group["trials"], f"{suite}: condition count")
        covered += len(part)
        rate_key = "support_success_rate" if "support_success_rate" in group else "exact_support_rate"
        close(support_rate(part), group[rate_key], 1e-14, f"{suite}: support rate")
        if "failures" in group:
            require(sum(int(row["failed"]) for row in part) == group["failures"], f"{suite}: exception count")
        for key, expected in group.items():
            if key.endswith("_median"):
                field = key[:-7]
                values = [float(row[field]) for row in part if row.get(field) not in (None, "")]
                require(bool(values), f"{suite}: missing median inputs")
                require(math.isclose(statistics.median(values), expected, rel_tol=1e-12, abs_tol=1e-14), f"{suite}: {key}")
    require(covered == len(rows), f"{suite}: summary condition coverage")


def verify_files(root):
    data = {}
    authenticated_sources = {}
    for suite, count in EXPECTED_ROWS.items():
        directory = root / "results" / suite
        manifest = json_file(directory / "manifest.json")
        require(manifest["suite"] == suite and manifest["rows"] == count, f"{suite}: manifest counts")
        for name, expected in manifest["data_sha256"].items():
            require(Path(name).name == name, "Invalid data-file identifier")
            require(sha256(directory / name) == expected, f"{suite}/{name}: data hash")
        for name, expected in manifest["distribution_source_hashes"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid source-file identifier")
            if name in authenticated_sources:
                require(authenticated_sources[name] == expected, "Inconsistent distribution source hashes")
            else:
                require(sha256(root / name) == expected, f"{name}: distributed source hash")
                authenticated_sources[name] = expected
        with (directory / "trials.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == count, f"{suite}: CSV row count")
        identity_fields = ("id", "method") if "id" in rows[0] else (
            ("N", "trial", "method") if suite == "validation" else
            ("total_photons_two_frames", "trial", "method") if suite == "case2d" else
            ("sparsity", "trial", "method"))
        require(len({tuple(row[key] for key in identity_fields) for row in rows}) == count, f"{suite}: duplicate identities")
        if "jobs" in manifest:
            require({row["id"] for row in rows} == {job["id"] for job in manifest["jobs"]}, f"{suite}: job coverage")
        verify_summary(rows, json_file(directory / "summary.json"), suite)
        data[suite] = rows
    frozen_path = root / "results/baseline_tuning/frozen.json"
    tuning_manifest = json_file(frozen_path.parent / "manifest.json")
    require(sha256(frozen_path) == tuning_manifest["frozen_sha256"], "Frozen settings hash")
    frozen = json_file(frozen_path)
    require(frozen["distribution_source_hashes"] == authenticated_sources, "Frozen distribution source identity")
    require(bool(frozen["recorded_solver_sources"]), "Missing recorded source provenance")
    for method in ("pie", "htwf", "htaf"):
        settings = frozen["baseline_configs"][method]
        require(settings["iterations"] == 1200 and settings["restarts"] == 32 and settings["threshold_after"] == 200, "Iterative baseline budget")
    require(frozen["baseline_configs"]["full_linear"] == {"iterations": 12, "restarts": 1}, "Full-linear budget")
    with np.load(root / "results/validation/raw_data.npz", allow_pickle=False) as arrays:
        require(set(arrays.files) == {f"N{n}_{field}" for n in (1, 2, 3) for field in ("clean_intensities", "supports", "amplitudes")}, "Validation array identities")
        for n in (1, 2, 3):
            require(arrays[f"N{n}_clean_intensities"].shape == (200, 2, 257), "Validation frame shape")
            require(arrays[f"N{n}_supports"].shape == arrays[f"N{n}_amplitudes"].shape == (200, n), "Validation truth shape")
    with np.load(root / "results/case2d/measured_intensities.npz", allow_pickle=False) as arrays:
        require(arrays["intensities"].shape == (1200, 2, 65, 65), "Two-dimensional frame shape")
        identities = {(float(total), int(trial), int(seed)) for total, trial, seed in zip(arrays["total_photons_two_frames"], arrays["trial"], arrays["seeds"])}
        require(len(identities) == 1200, "Two-dimensional input identities")
        require(identities == {(float(row["total_photons_two_frames"]), int(row["trial"]), int(row["seed"])) for row in data["case2d"]}, "Two-dimensional frame/CSV pairing")
    return data


def verify_paper(data):
    validation = data["validation"]
    for n in (1, 2, 3):
        close(support_rate(select(validation, N=n, method="joint_global")), 1, 0, "Noiseless joint support")
        close(support_rate(select(validation, N=n, method="spectral_algebraic")), 1 if n < 3 else 29 / 200, 0, "Spectral support")
    gaussian = data["gaussian"]
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
    print("Data hashes, distributed source hashes, summaries, array pairing, and manuscript values agree.")
    print("Recorded source provenance is separate from the distributed implementation; timings are not rerun.")


if __name__ == "__main__":
    main()
