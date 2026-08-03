#!/usr/bin/env python3
"""Run deterministic and small Monte Carlo validation on the local machine."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from twoscan_ptycho.core import (
    affine_object,
    clean_lag_moments,
    diffraction_intensity,
    gaussian_probe,
    moment_dictionaries,
    random_amplitudes,
    recover_from_two_intensities,
    reflect_sparse_object,
    sample_separated_positions,
    sparse_perturbation,
    split_null_counterexample,
    support_metrics,
    theoretical_moments,
    wrap_centered,
)


def orbit_support(positions: tuple[int, ...], scan: int, n: int) -> tuple[int, ...]:
    return tuple(sorted(set(abs(wrap_centered(t - scan, n)) for t in positions)))


def condition_on_support(
    gamma: float, lag_count: int, max_offset: int, support: tuple[int, ...]
) -> tuple[float, float]:
    c, s = moment_dictionaries(gamma, lag_count, max_offset)
    c_cond = float(np.linalg.cond(c[:, support]))
    positive = tuple(x for x in support if x > 0)
    s_cond = (
        float(np.linalg.cond(s[:, np.asarray(positive) - 1])) if positive else 1.0
    )
    return c_cond, s_cond


def run_case(
    n_atoms: int,
    trial: int,
    rng: np.random.Generator,
    beam_width: int,
) -> dict[str, object]:
    n = 257
    scans = (0, 1)
    lag_count = 2 * n_atoms
    location_bound = max(
        10, int(np.ceil((n_atoms - 1) * (lag_count + 2) / 2.0)) + 5
    )
    gamma = 3.0 / (location_bound * location_bound)
    positions = sample_separated_positions(
        n,
        n_atoms,
        minimum_distance=lag_count,
        scans=scans,
        margin=lag_count,
        rng=rng,
        location_bound=location_bound,
    )
    amplitudes = random_amplitudes(n_atoms, rng, physical_phase=True)
    probe = gaussian_probe(n, gamma)
    z = affine_object(n, positions, amplitudes)
    intensities = tuple(diffraction_intensity(z, probe, scan) for scan in scans)
    max_offset = location_bound + max(abs(scan) for scan in scans) + 1

    formula_errors = []
    reflection_errors = []
    conditions = []
    for scan, intensity in zip(scans, intensities):
        measured = clean_lag_moments(intensity, probe, gamma, lag_count)
        predicted = theoretical_moments(
            positions, amplitudes, scan, gamma, lag_count, n
        )
        formula_errors.append(
            float(
                np.linalg.norm(measured - predicted)
                / max(np.linalg.norm(predicted), np.finfo(float).eps)
            )
        )
        reflected_positions, reflected_amplitudes = reflect_sparse_object(
            n, positions, amplitudes, scan
        )
        reflected_intensity = diffraction_intensity(
            affine_object(n, reflected_positions, reflected_amplitudes), probe, scan
        )
        reflection_errors.append(
            float(
                np.linalg.norm(intensity - reflected_intensity)
                / max(np.linalg.norm(intensity), np.finfo(float).eps)
            )
        )
        support = orbit_support(positions, scan, n)
        conditions.append(
            condition_on_support(gamma, lag_count, max_offset, support)
        )

    truth = sparse_perturbation(n, positions, amplitudes)
    beam = recover_from_two_intensities(
        intensities[0],
        intensities[1],
        probe,
        gamma,
        scans[0],
        scans[1],
        n_atoms,
        max_offset,
        lag_count=lag_count,
        solver="beam",
        beam_width=beam_width,
    )
    omp = recover_from_two_intensities(
        intensities[0],
        intensities[1],
        probe,
        gamma,
        scans[0],
        scans[1],
        n_atoms,
        max_offset,
        lag_count=lag_count,
        solver="omp",
    )
    beam_metrics = support_metrics(truth, beam.result.perturbation, n_atoms)
    omp_metrics = support_metrics(truth, omp.result.perturbation, n_atoms)
    return {
        "n_atoms": n_atoms,
        "trial": trial,
        "n": n,
        "lag_count": lag_count,
        "location_bound": location_bound,
        "gamma": gamma,
        "positions": positions,
        "amplitudes": [[a.real, a.imag] for a in amplitudes],
        "formula_relative_error_max": max(formula_errors),
        "one_scan_reflection_relative_error_max": max(reflection_errors),
        "condition_real_max": max(value[0] for value in conditions),
        "condition_imag_max": max(value[1] for value in conditions),
        "beam": {
            **beam_metrics,
            "scan_supports": [list(item.support) for item in beam.scan_recoveries],
            "scan_residuals": [item.residual_norm for item in beam.scan_recoveries],
        },
        "omp": {
            **omp_metrics,
            "scan_supports": [list(item.support) for item in omp.scan_recoveries],
            "scan_residuals": [item.residual_norm for item in omp.scan_recoveries],
        },
    }


def summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "formula_relative_error_max": max(
            float(case["formula_relative_error_max"]) for case in cases
        ),
        "one_scan_reflection_relative_error_max": max(
            float(case["one_scan_reflection_relative_error_max"]) for case in cases
        ),
    }
    by_n: dict[str, object] = {}
    for n_atoms in sorted(set(int(case["n_atoms"]) for case in cases)):
        group = [case for case in cases if int(case["n_atoms"]) == n_atoms]
        by_n[str(n_atoms)] = {
            "trials": len(group),
            "beam_support_success_rate": float(
                np.mean([float(case["beam"]["support_exact"]) for case in group])
            ),
            "beam_nrmse_median": float(
                np.median([float(case["beam"]["nrmse"]) for case in group])
            ),
            "omp_support_success_rate": float(
                np.mean([float(case["omp"]["support_exact"]) for case in group])
            ),
            "omp_nrmse_median": float(
                np.median([float(case["omp"]["nrmse"]) for case in group])
            ),
            "condition_real_median": float(
                np.median([float(case["condition_real_max"]) for case in group])
            ),
            "condition_imag_median": float(
                np.median([float(case["condition_imag_max"]) for case in group])
            ),
        }
    summary["by_n_atoms"] = by_n
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-atoms", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/validation/summary.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cases = [
        run_case(n_atoms, trial, rng, args.beam_width)
        for n_atoms in range(1, args.max_atoms + 1)
        for trial in range(args.trials)
    ]
    null_examples = {
        str(n_atoms): {
            "lag_count": 2 * n_atoms - 1,
            "relative_residual": float(
                split_null_counterexample(
                    gamma=0.01 / max(n_atoms, 1), sparsity=n_atoms
                )["relative_residual"]
            ),
        }
        for n_atoms in range(1, args.max_atoms + 1)
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "configuration": {
            "trials_per_n": args.trials,
            "max_atoms": args.max_atoms,
            "beam_width": args.beam_width,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "summary": summarize(cases),
        "two_n_minus_one_null_examples": null_examples,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
