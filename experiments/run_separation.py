#!/usr/bin/env python3
"""Noiseless sweep across the short-lag separation boundary."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from twoscan_ptycho.core import (
    affine_object,
    clean_lag_moments,
    diffraction_intensity,
    gaussian_probe,
    random_amplitudes,
    recover_from_two_intensities,
    sparse_perturbation,
    support_metrics,
    theoretical_moments,
)


def equally_spaced_positions(n_atoms: int, spacing: int) -> tuple[int, ...]:
    raw = np.arange(n_atoms, dtype=int) * spacing
    raw = raw - int(np.round(np.mean(raw)))
    return tuple(int(value) for value in raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--atoms", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--beam-width", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/separation"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = 509
    scans = (0, 1)
    rows: list[dict[str, object]] = []

    for n_atoms in args.atoms:
        lag_count = 2 * n_atoms
        for spacing in range(1, 3 * lag_count + 1):
            positions = equally_spaced_positions(n_atoms, spacing)
            max_offset = max(
                abs(position - scan) for position in positions for scan in scans
            ) + 2
            gamma = 5.0 / max(max_offset * max_offset, 1)
            probe = gaussian_probe(n, gamma)
            for trial in range(args.trials):
                amplitudes = random_amplitudes(
                    n_atoms,
                    rng,
                    physical_phase=True,
                    minimum_magnitude=0.75,
                )
                z = affine_object(n, positions, amplitudes)
                intensities = tuple(
                    diffraction_intensity(z, probe, scan) for scan in scans
                )
                formula_errors = []
                for scan, intensity in zip(scans, intensities):
                    measured = clean_lag_moments(
                        intensity, probe, gamma, lag_count
                    )
                    predicted = theoretical_moments(
                        positions,
                        amplitudes,
                        scan,
                        gamma,
                        lag_count,
                        n,
                    )
                    formula_errors.append(
                        float(
                            np.linalg.norm(measured - predicted)
                            / max(np.linalg.norm(predicted), np.finfo(float).eps)
                        )
                    )
                recovered = recover_from_two_intensities(
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
                    beam_width=args.beam_width,
                    row_scaling="dictionary",
                )
                metrics = support_metrics(
                    sparse_perturbation(n, positions, amplitudes),
                    recovered.result.perturbation,
                    n_atoms,
                )
                rows.append(
                    {
                        "n_atoms": n_atoms,
                        "lag_count": lag_count,
                        "spacing": spacing,
                        "spacing_over_2n": spacing / lag_count,
                        "trial": trial,
                        "formula_relative_error": max(formula_errors),
                        **metrics,
                    }
                )

    summaries: list[dict[str, object]] = []
    for n_atoms in args.atoms:
        for spacing in range(1, 6 * n_atoms + 1):
            group = [
                row
                for row in rows
                if int(row["n_atoms"]) == n_atoms
                and int(row["spacing"]) == spacing
            ]
            summaries.append(
                {
                    "n_atoms": n_atoms,
                    "lag_count": 2 * n_atoms,
                    "spacing": spacing,
                    "spacing_over_2n": spacing / (2 * n_atoms),
                    "formula_error_median": float(
                        np.median(
                            [float(row["formula_relative_error"]) for row in group]
                        )
                    ),
                    "support_success_rate": float(
                        np.mean([float(row["support_exact"]) for row in group])
                    ),
                    "nrmse_median": float(
                        np.median([float(row["nrmse"]) for row in group])
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
