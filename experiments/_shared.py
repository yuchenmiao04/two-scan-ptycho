"""Serialization and protocol helpers for the reported auxiliary experiments."""
from __future__ import annotations

import os

for _variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes():
    paths = sorted((ROOT / "src/twoscan_ptycho").glob("*.py"))
    paths += sorted((ROOT / "experiments").glob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _machine():
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {
        "cpu": cpu, "logical_cpu_count": os.cpu_count(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__,
        "blas_thread_environment": {name: os.environ[name] for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        )},
    }


def _manifest(output, suite, args, settings):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite or mix an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    value = {
        "suite": suite, "seed": args.seed,
        "source_sha256": _source_hashes(), "machine": _machine(), "settings": settings,
        "smoke": args.smoke,
        "runtime_interpretation": "Diagnostic wall times are not controlled serial benchmark runtimes.",
    }
    _write_json(output / "manifest.json", value)
    return value


def _number_summary(values):
    array = np.asarray(values, float)
    finite = array[np.isfinite(array)]
    return {
        "count": len(array), "nonfinite_count": int(np.sum(~np.isfinite(array))),
        "min": float(np.min(finite)) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
        "q25_q75": np.quantile(finite, [.25, .75]).tolist() if len(finite) else None,
    }


def _support_string(support):
    return json.dumps(support, separators=(",", ":"))


def _legal_support(rng, count, bound):
    """Uniform line-separated support; settings also ensure cyclic spacing."""
    separation = 2 * count
    available = 2 * bound + 1 - separation * (count - 1)
    if available < count:
        raise ValueError("Field of view is too small for the requested support.")
    compressed = np.sort(rng.choice(available, size=count, replace=False))
    return tuple(int(x) for x in compressed + separation * np.arange(count) - bound)


def _coefficient_vector(model, support, amplitudes):
    k = len(model.candidates)
    vector = np.zeros(2 * k)
    for t, amplitude in zip(support, amplitudes):
        j = model.candidates.index(t)
        vector[j] = amplitude.real
        vector[k + j] = amplitude.imag
    return vector


def _wilson(success, count):
    z = 1.959963984540054
    rate = success / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - half), min(1., center + half)]


def positive_integer(value):
    import argparse
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number
