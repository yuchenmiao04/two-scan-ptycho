"""Generate the numerical trace used in Fig. 1(a), without plotting it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from twoscan_ptycho.core import (
    affine_object,
    diffraction_intensity,
    gaussian_probe,
    intensity_autocorrelation,
    theoretical_moments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/validation")
    )
    args = parser.parse_args()

    n = 257
    sparsity = 3
    gamma = 0.02
    scan = 0
    positions = (-12, 0, 12)
    amplitudes = np.exp(1j * np.asarray((0.65, -1.10, 1.45))) - 1.0
    probe = gaussian_probe(n, gamma)
    z = affine_object(n, positions, amplitudes)
    intensity = diffraction_intensity(z, probe, scan)
    background = np.abs(np.fft.fft(probe)) ** 2
    observed = intensity_autocorrelation(intensity) - intensity_autocorrelation(
        background
    )
    lags = np.arange(1, 19)
    predicted = np.exp(-gamma * lags**2) * theoretical_moments(
        positions, amplitudes, scan, gamma, int(lags[-1]), n
    )
    scale = float(np.max(np.abs(observed[lags])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "clean_lag_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "lag",
                "observed_real",
                "observed_imag",
                "predicted_real",
                "predicted_imag",
                "observed_magnitude_normalized",
                "predicted_magnitude_normalized",
                "is_clean_lag",
            ),
        )
        writer.writeheader()
        for index, lag in enumerate(lags):
            writer.writerow(
                {
                    "lag": int(lag),
                    "observed_real": float(np.real(observed[lag])),
                    "observed_imag": float(np.imag(observed[lag])),
                    "predicted_real": float(np.real(predicted[index])),
                    "predicted_imag": float(np.imag(predicted[index])),
                    "observed_magnitude_normalized": float(
                        np.abs(observed[lag]) / scale
                    ),
                    "predicted_magnitude_normalized": float(
                        np.abs(predicted[index]) / scale
                    ),
                    "is_clean_lag": int(lag <= 2 * sparsity),
                }
            )

    metadata = {
        "figure_panel": "Fig. 1(a)",
        "n": n,
        "sparsity": sparsity,
        "gamma": gamma,
        "scan": scan,
        "positions": positions,
        "phases_rad": (0.65, -1.10, 1.45),
        "clean_lag_limit": 2 * sparsity,
        "normalization": scale,
    }
    (args.output_dir / "clean_lag_trace.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
