"""Serial timing of fresh joint dictionary and SVD setup plus reconstruction."""
from __future__ import annotations

from experiments.run_experiments import (
    ROOT, SEEDS, calculate_metrics, configuration, dump_json, hardware,
    make_instance, source_hashes as production_source_hashes, stable_seed,
)
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import time
import numpy as np

from twoscan_ptycho.reconstruction import GlobalSupportSolver, build_joint_dictionary

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
        "note": "Fresh setup measurements from this invocation; archived measurements are not overwritten.",
    })
    return output


def write_rows(output, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/timing_setup")
    parser.add_argument("--trials", type=positive_integer, default=30)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--namespace", type=int, default=SEEDS["timing"])
    parser.add_argument("--sparsities", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--snr", choices=("40", "60", "80", "inf"), default="60")
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--confirm-isolated", action="store_true",
        help="Operator declaration that competing experiments are stopped; not independently verified.",
    )
    args = parser.parse_args()
    if args.trial_start < 0:
        parser.error("--trial-start must be nonnegative")
    run_timing_setup(args)


if __name__ == "__main__":
    main()
