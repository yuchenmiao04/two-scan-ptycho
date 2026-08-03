"""Run the two-dimensional calibrated phase-defect case study."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path

import numpy as np

from twoscan_ptycho.case2d import (
    add_poisson_noise_2d,
    build_calibrated_signature_model_2d,
    diffraction_intensity_2d,
    gaussian_probe_2d,
    profiled_support_score_2d,
    calibrated_signature_target_2d,
    recover_calibrated_case_2d,
    sparse_phase_object_2d,
    support_exact,
)


N = 65
SPARSITY = 3
GAMMA = 0.015
LAG_COUNT = 12
OBJECT_BOUND = 10
SCANS = ((-2, -1), (3, 2))
TRUE_POSITIONS = ((-9, -8), (0, 8), (8, -7))
DEFECT_PHASE = 0.8


def _parse_photons(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def _format_positions(positions: tuple[tuple[int, int], ...]) -> str:
    return ";".join(f"{p1}:{p2}" for p1, p2 in positions)


def _localization_rmse(
    truth: tuple[tuple[int, int], ...], estimate: tuple[tuple[int, int], ...]
) -> float:
    best = np.inf
    for ordering in permutations(estimate):
        squared = sum(
            (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2
            for left, right in zip(truth, ordering)
        )
        best = min(best, squared)
    return float(np.sqrt(best / len(truth)))


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    defect_signature = np.exp(1j * DEFECT_PHASE) - 1.0
    probe = gaussian_probe_2d(N, GAMMA)
    object_array, _ = sparse_phase_object_2d(
        N, TRUE_POSITIONS, [DEFECT_PHASE] * SPARSITY
    )
    clean_intensities = tuple(
        diffraction_intensity_2d(object_array, probe, scan) for scan in SCANS
    )
    calibrated = build_calibrated_signature_model_2d(
        SCANS,
        GAMMA,
        LAG_COUNT,
        N,
        OBJECT_BOUND,
        defect_signature,
        background_gamma=GAMMA,
    )
    mismatched = build_calibrated_signature_model_2d(
        SCANS,
        GAMMA * (1.0 + args.mismatch),
        LAG_COUNT,
        N,
        OBJECT_BOUND,
        defect_signature,
        background_gamma=GAMMA,
    )

    rows: list[dict[str, object]] = []
    aggregate = []
    for photon_index, photons in enumerate(args.photons):
        successes = {"calibrated": 0, "mismatched": 0}
        localization = {"calibrated": [], "mismatched": []}
        for trial in range(args.trials):
            rng = np.random.default_rng(
                args.seed + 100_003 * photon_index + 1_009 * trial
            )
            noisy = tuple(
                add_poisson_noise_2d(values, photons, rng)
                for values in clean_intensities
            )
            results = {
                "calibrated": recover_calibrated_case_2d(
                    noisy, calibrated, SPARSITY, refinement_sweeps=3
                ),
                "mismatched": recover_calibrated_case_2d(
                    noisy, mismatched, SPARSITY, refinement_sweeps=3
                ),
            }
            row: dict[str, object] = {
                "photons_per_scan": photons,
                "trial": trial,
            }
            for label, result in results.items():
                exact = support_exact(TRUE_POSITIONS, result.positions)
                error = _localization_rmse(TRUE_POSITIONS, result.positions)
                successes[label] += int(exact)
                localization[label].append(error)
                row[f"{label}_success"] = int(exact)
                row[f"{label}_localization_rmse"] = error
                row[f"{label}_positions"] = _format_positions(result.positions)
                row[f"{label}_relative_residual"] = result.relative_residual
            rows.append(row)
        item = {
            "photons_per_scan": photons,
            "calibrated_success_rate": successes["calibrated"] / args.trials,
            "mismatched_success_rate": successes["mismatched"] / args.trials,
            "calibrated_median_localization_rmse": float(
                np.median(localization["calibrated"])
            ),
            "mismatched_median_localization_rmse": float(
                np.median(localization["mismatched"])
            ),
        }
        aggregate.append(item)
        print(
            f"photons={photons:.3g}: calibrated="
            f"{100.0 * item['calibrated_success_rate']:.1f}%, mismatch="
            f"{100.0 * item['mismatched_success_rate']:.1f}%",
            flush=True,
        )

    fixed_rng = np.random.default_rng(args.seed)
    fixed_noisy = tuple(
        add_poisson_noise_2d(values, args.fixed_photons, fixed_rng)
        for values in clean_intensities
    )
    fixed = recover_calibrated_case_2d(
        fixed_noisy, calibrated, SPARSITY, refinement_sweeps=3
    )
    fixed_target = calibrated_signature_target_2d(fixed_noisy, calibrated)
    profile_score = profiled_support_score_2d(
        fixed_target, calibrated, fixed.positions
    )
    np.savez_compressed(
        output / "fixed_case.npz",
        true_positions=np.asarray(TRUE_POSITIONS, dtype=int),
        recovered_positions=np.asarray(fixed.positions, dtype=int),
        recovered_strengths=np.asarray(fixed.strengths, dtype=float),
        scan_centers=np.asarray(SCANS, dtype=int),
        defect_phase=DEFECT_PHASE,
        photons_per_scan=args.fixed_photons,
        intensity_scan_1=fixed_noisy[0],
        intensity_scan_2=fixed_noisy[1],
        profiled_support_score_db=profile_score,
    )
    with (output / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "seed": args.seed,
        "trials_per_point": args.trials,
        "n": N,
        "sparsity": SPARSITY,
        "gamma": GAMMA,
        "lag_box": [LAG_COUNT, LAG_COUNT],
        "scan_centers": SCANS,
        "true_positions": TRUE_POSITIONS,
        "defect_phase_rad": DEFECT_PHASE,
        "probe_width_mismatch_fraction": args.mismatch,
        "blank_field_background_subtraction": True,
        "aggregate": aggregate,
        "fixed_case": {
            "photons_per_scan": args.fixed_photons,
            "recovered_positions": fixed.positions,
            "recovered_strengths": fixed.strengths,
            "exact_support": support_exact(TRUE_POSITIONS, fixed.positions),
            "relative_residual": fixed.relative_residual,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--photons",
        type=_parse_photons,
        default=_parse_photons("3e7,1e8,3e8,1e9,3e9,1e10"),
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--mismatch", type=float, default=0.02)
    parser.add_argument("--fixed-photons", type=float, default=1e9)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output", default="results/poisson_2d"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
