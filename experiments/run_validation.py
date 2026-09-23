"""Noiseless joint and spectral validation: 200 paired objects per sparsity."""
from __future__ import annotations

from experiments._shared import (
    ROOT, _manifest, _write_csv, _write_json, _number_summary, _support_string,
    _legal_support, _coefficient_vector, positive_integer,
)
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from twoscan_ptycho.core import affine_object, diffraction_intensity
from twoscan_ptycho.reconstruction import (
    GlobalSupportSolver, SpectralConfig, build_joint_dictionary, extract_raw_data,
    recover_two_scan_spectral,
)

def run_validation(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trials = 3 if args.smoke else args.trials
    settings = {"n": 257, "scans": [0, 1], "sparsities": [1, 2, 3], "candidate_counts": [21, 21, 27], "trials_per_N": trials, "gamma": "5/B^2", "strict_separation": ">2N", "amplitudes": "magnitude uniform[0.5,1.5], phase uniform[-pi,pi]", "noise": "none", "spectral_tolerances": vars(SpectralConfig())}
    _manifest(output, "validation", args, settings)
    rows, archived, aggregate = [], {}, []
    for count, bound in ((1, 10), (2, 10), (3, 13)):
        gamma = 5 / bound ** 2
        model = build_joint_dictionary(257, gamma, (0, 1), range(-bound, bound + 1), count)
        joint = GlobalSupportSolver(model, count)
        stored_frames, stored_support, stored_amplitudes = [], [], []
        for trial in range(trials):
            seed = args.seed + 10_000_019 + count * 100_003 + trial * 1_009
            rng = np.random.default_rng(seed)
            support = _legal_support(rng, count, bound)
            amplitudes = rng.uniform(.5, 1.5, count) * np.exp(1j * rng.uniform(-np.pi, np.pi, count))
            z = affine_object(257, support, amplitudes)
            frames = np.array([diffraction_intensity(z, model.probes[i], r) for i, r in enumerate(model.scans)])
            raw = extract_raw_data(frames, model)
            identity = np.linalg.norm(raw - model.matrix @ _coefficient_vector(model, support, amplitudes)) / max(np.linalg.norm(raw), np.finfo(float).tiny)
            results = {
                "joint_global": joint.solve_intensities(frames),
                "spectral_algebraic": recover_two_scan_spectral(frames, model.probes[0], gamma, model.scans, count),
            }
            for name, result in results.items():
                rows.append({
                    "N": count, "trial": trial, "seed": seed, "method": name, "n": 257,
                    "candidate_count": 2 * bound + 1, "gamma": gamma,
                    "status": result.status, "algorithm_passed": int(result.status.startswith("ok_")),
                    "support_exact": int(set(result.support) == set(support)),
                    "perturbation_nrmse": float(np.linalg.norm(result.u - (z - 1)) / np.linalg.norm(z - 1)),
                    "raw_clean_lag_identity_relative_error": identity,
                    "raw_residual": result.residual, "runtime_seconds_diagnostic": result.timings["total_online"],
                    "true_support": _support_string(support), "estimated_support": _support_string(result.support),
                    "amplitudes_real": _support_string(amplitudes.real.tolist()), "amplitudes_imag": _support_string(amplitudes.imag.tolist()),
                })
            stored_frames.append(frames)
            stored_support.append(support)
            stored_amplitudes.append(amplitudes)
        archived[f"N{count}_clean_intensities"] = np.asarray(stored_frames)
        archived[f"N{count}_supports"] = np.asarray(stored_support)
        archived[f"N{count}_amplitudes"] = np.asarray(stored_amplitudes)
        for method in results:
            part = [row for row in rows if row["N"] == count and row["method"] == method]
            passed = [row for row in part if row["algorithm_passed"]]
            aggregate.append({
                "N": count, "method": method, "trials": trials,
                "algorithm_pass_rate": sum(row["algorithm_passed"] for row in part) / trials,
                "exact_support_rate": sum(row["support_exact"] for row in part) / trials,
                "status_counts": dict(Counter(row["status"] for row in part)),
                "nrmse_all_returned_estimates": _number_summary([row["perturbation_nrmse"] for row in part]),
                "nrmse_when_algorithm_passed": _number_summary([row["perturbation_nrmse"] for row in passed]),
                "identity_error": _number_summary([row["raw_clean_lag_identity_relative_error"] for row in part]),
            })
        print(f"validation N={count}: " + "; ".join(f"{item['method']} pass={item['algorithm_pass_rate']:.3f}, support={item['exact_support_rate']:.3f}" for item in aggregate if item["N"] == count), flush=True)
    _write_csv(output / "trials.csv", rows)
    np.savez_compressed(output / "raw_data.npz", **archived)
    summary = {"settings": settings, "aggregate": aggregate, "failure_interpretation": "Spectral failures return a zero object; their NRMSE is retained and reported separately from passed estimates. Tolerances are fixed before all trials."}
    _write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/validation")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--trials", type=positive_integer, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.output = args.output / "smoke"
    run_validation(args)


if __name__ == "__main__":
    main()
