"""Raw-lag conditioning across probe widths and complete N=2 support unions."""
from __future__ import annotations
from experiments._shared import (
    ROOT, _manifest, _write_json, _write_csv, _number_summary,
    _support_string, _legal_support, positive_integer,
)
import argparse
from itertools import combinations_with_replacement
from math import ceil
from pathlib import Path
import numpy as np
from twoscan_ptycho.reconstruction import build_joint_dictionary, GlobalSupportSolver
WIDTH_FACTORS = (.25, .5, 1., 2., 4.)

def _union_singular(model, union):
    k = len(model.candidates)
    indices = [model.candidates.index(t) for t in union]
    singular = np.linalg.svd(model.matrix[:, indices + [j + k for j in indices]], compute_uv=False)
    gain = float(singular[-1])
    condition = float(singular[0] / gain) if gain > 0 else float("inf")
    return gain, condition, singular



def _complete_beta_N2(output):
    model = build_joint_dictionary(257, .05, (0, 1), range(-10, 11), 2)
    solver = GlobalSupportSolver(model, 2)
    unions = sorted({tuple(sorted(set(a) | set(b))) for a, b in combinations_with_replacement(solver.all_supports, 2) if a or b}, key=lambda x: (len(x), x))
    gains, conditions = [], []
    singular_values = np.full((len(unions), 8), np.nan)
    encoded = np.full((len(unions), 4), -999, dtype=int)
    for j, union in enumerate(unions):
        gain, condition, singular = _union_singular(model, union)
        gains.append(gain)
        conditions.append(condition)
        singular_values[j, :len(singular)] = singular
        encoded[j, :len(union)] = union
    minimum = int(np.argmin(gains))
    np.savez_compressed(output / "beta_N2_all_support_unions.npz", matrix=model.matrix, candidate_positions=model.candidates, union_positions=encoded, union_sizes=[len(x) for x in unions], gains=gains, conditions=conditions, singular_values=singular_values)
    return {
        "N": 2, "n": 257, "K": 21, "gamma": .05, "scans": [0, 1],
        "legal_support_count_including_empty": solver.candidate_count,
        "support_pairs_including_empty_pair": solver.candidate_count * (solver.candidate_count + 1) // 2,
        "distinct_nonempty_unions": len(unions), "beta_numerical_complete_enumeration": gains[minimum],
        "minimizing_union": unions[minimum], "minimizing_union_condition": conditions[minimum],
        "normalization": "Original raw-lag operator; physical Re/Im coefficient norm; no column normalization.",
        "qualification": "Every admissible union is enumerated, but its singular values are ordinary floating-point evaluations, not validated interval lower bounds.",
    }



def run_conditioning(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trials = 3 if args.smoke else args.trials
    counts = (1, 2, 3) if args.smoke else tuple(range(1, 9))
    settings = {"n": 257, "scans": [0, 1], "N": counts, "width_factors": WIDTH_FACTORS, "base_gamma": "5/B^2", "B": "max(10,ceil((N-1)*(2N+2)/2)+5)", "support_pairs_per_point": trials, "support_pairs_shared_across_widths": True, "normalization": "raw dictionary without row or column normalization"}
    _manifest(output, "conditioning", args, settings)
    rows, archive, aggregate = [], {}, []
    for count in counts:
        bound = max(10, ceil((count - 1) * (2 * count + 2) / 2) + 5)
        support_pairs = []
        for trial in range(trials):
            seed = args.seed + 20_000_033 + count * 100_003 + trial * 1_009
            rng = np.random.default_rng(seed)
            support_pairs.append((seed, _legal_support(rng, count, bound), _legal_support(rng, count, bound)))
        for factor in WIDTH_FACTORS:
            gamma = factor * 5 / bound ** 2
            model = build_joint_dictionary(257, gamma, (0, 1), range(-bound, bound + 1), count)
            singular_array = np.full((trials, 4 * count), np.nan)
            key = f"N{count}_factor{factor:g}"
            archive[key + "_matrix"] = model.matrix
            for trial, (seed, left, right) in enumerate(support_pairs):
                union = tuple(sorted(set(left) | set(right)))
                gain, condition, singular = _union_singular(model, union)
                singular_array[trial, :len(singular)] = singular
                rows.append({"N": count, "width_factor": factor, "gamma": gamma, "B": bound, "trial": trial, "seed": seed, "left_support": _support_string(left), "right_support": _support_string(right), "union_support": _support_string(union), "union_size": len(union), "minimum_gain": gain, "maximum_gain": float(singular[0]), "condition": condition, "log10_minimum_gain": float(np.log10(gain)) if gain > 0 else float("-inf"), "log10_condition": float(np.log10(condition)) if condition > 0 else float("inf")})
            archive[key + "_singular_values"] = singular_array
            part = [row for row in rows if row["N"] == count and row["width_factor"] == factor]
            aggregate.append({"N": count, "width_factor": factor, "gamma": gamma, "B": bound, "trials": trials, "minimum_gain": _number_summary([row["minimum_gain"] for row in part]), "condition": _number_summary([row["condition"] for row in part]), "log10_condition": _number_summary([row["log10_condition"] for row in part])})
        print(f"conditioning N={count}, B={bound}: five widths completed", flush=True)
    exact_beta = _complete_beta_N2(output)
    print(f"conditioning exact N2: beta={exact_beta['beta_numerical_complete_enumeration']:.6g}, {exact_beta['distinct_nonempty_unions']} unions", flush=True)
    _write_csv(output / "trials.csv", rows)
    np.savez_compressed(output / "raw_dictionaries_and_singular_values.npz", **archive)
    summary = {"settings": settings, "aggregate": aggregate, "exact_N2_beta": exact_beta, "qualification": "Random-pair minimum gains are sampled values, not uniform lower bounds. Only the separate N2 calculation enumerates every legal union, and even it uses unvalidated floating-point SVD."}
    _write_json(output / "summary.json", summary)
    return summary



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/conditioning")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--trials", type=positive_integer, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.output = args.output / "smoke"
    run_conditioning(args)


if __name__ == "__main__":
    main()
