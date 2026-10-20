"""Paired serial N=2, 80-dB timing with the frozen methods.

Run from the repository root::

    python3 -m experiments.run_timing_80db

The standard timing namespace and trial indices pair these objects with the
20/60/noiseless timing suite. A separate cold-setup run is needed for the
joint method's fresh dictionary-plus-SVD cost.
"""
from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.run_experiments import (
    ROOT, SEEDS, dump_json, execute, perform_job, source_hashes, validate_frozen_sources,
)


def distribution(values, expected_count):
    data = np.asarray(values, dtype=float)
    if len(data) != expected_count or not np.all(np.isfinite(data)):
        raise ValueError(f"Expected exactly {expected_count} finite paired timing observations")
    return {
        "count": len(data),
        "median": float(np.median(data)),
        "q25_q75": [float(x) for x in np.quantile(data, [0.25, 0.75])],
        "q10_q90": [float(x) for x in np.quantile(data, [0.1, 0.9])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/timing_80db")
    parser.add_argument("--frozen", type=Path, default=ROOT / "results/baseline_tuning/frozen.json")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new or empty output directory")
    frozen_path = args.frozen
    output = args.output / "paired_online"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_frozen_sources(frozen)
    methods = ("joint", "full_linear")
    config = frozen["baseline_configs"]["full_linear"]
    jobs = [
        dict(suite="timing", namespace=SEEDS["timing"], trial=trial,
             sparsity=2, snr=80, methods=methods,
             baseline_configs={"full_linear": config})
        for trial in range(args.trials)
    ]
    # Same one unreported warm-up convention as run_experiments.py's timing suite.
    perform_job(dict(jobs[0], trial=1_000_000, id="warmup"))
    wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = execute(jobs, output, workers=1, metadata={
        "frozen_file": frozen_path.name,
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "wrapper_sha256": wrapper_hash,
        "warmups": 1,
        "scope": "paired joint LS and full-linear, N=2, target defect-intensity SNR=80 dB",
        "timing_definition": "Online excludes dictionary/SVD setup; full-linear setup+online is also reported. Fresh dictionary and support-factorization cost is measured by the separate setup benchmark.",
    })
    if len(rows) != 2 * args.trials or any(row["failed"] for row in rows):
        raise RuntimeError("Expected two successful method records per paired trial")
    paired = {}
    for row in rows:
        paired.setdefault(int(row["trial"]), {})[row["method"]] = row
    if len(paired) != args.trials:
        raise RuntimeError("Missing paired trials")
    for pair in paired.values():
        if set(pair) != set(methods):
            raise RuntimeError("Missing a method in a paired trial")
        for key in ("true_support", "effective_snr_db", "total_snr_db"):
            if pair["joint"][key] != pair["full_linear"][key]:
                raise RuntimeError(f"Methods did not share the same {key}")
    by_method = {method: [paired[t][method] for t in sorted(paired)] for method in methods}
    output_summary = {
        "scope": "N=2, 80-dB target defect-intensity SNR, serial execution",
        "methods": methods,
        "trials_per_method": args.trials,
        "source_rows": "paired_online/trials.csv",
        "units": "milliseconds",
        "online_ms": {method: distribution([1e3 * row["online_seconds"] for row in group], args.trials)
                      for method, group in by_method.items()},
        "full_linear_setup_plus_online_ms": distribution([
            1e3 * row["algorithm_total_seconds"] for row in by_method["full_linear"]], args.trials),
        "full_linear_setup_ms": distribution([
            1e3 * row["setup_seconds"] for row in by_method["full_linear"]], args.trials),
        "paired_online_speedup": distribution([
            paired[t]["full_linear"]["online_seconds"] / paired[t]["joint"]["online_seconds"]
            for t in sorted(paired)], args.trials),
        "support_successes": {method: sum(int(row["support_exact"]) for row in group)
                              for method, group in by_method.items()},
        "note": "The joint setup_seconds in the production online runner is a cached solver attribute and is not a fresh per-trial setup measurement. Use experiments.run_timing_setup for fresh dictionary and support-factorization measurements.",
    }
    dump_json(output.parent / "paired_quantiles.json", output_summary)
    print(json.dumps(output_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
