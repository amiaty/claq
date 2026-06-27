#!/usr/bin/env python3
"""Run the complete CLAQ pipeline on a small synthetic archive."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claq.precomputed import load_precomputed_dataset
from make_synthetic_dataset import create_synthetic_dataset
import run_precomputed_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/claq_quickstart"),
        help="Directory for the generated archive and run outputs",
    )
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    data_path = create_synthetic_dataset(work_dir / "synthetic.npz")
    dataset = load_precomputed_dataset(data_path)
    print(
        f"Synthetic archive: {data_path} | train={len(dataset.train.labels)} "
        f"validation={len(dataset.validation.labels)} test={len(dataset.test.labels)} "
        f"queries={dataset.num_queries}"
    )

    original_argv = sys.argv
    try:
        sys.argv = [
            "run_precomputed_experiment.py",
            "--data", str(data_path),
            "--output", str(work_dir / "run"),
            "--variant-name", "CLAQ",
            "--lambdas", "0.3",
            "--seeds", "0",
            "--budget", "2",
            "--epochs", "1",
            "--batch-size", "32",
            "--learning-rate", "1e-3",
            "--hidden-dims", "32,16",
            "--device", "cpu",
            "--deterministic",
            "--overwrite",
        ]
        run_precomputed_experiment.main()
    finally:
        sys.argv = original_argv

    summary = work_dir / "run" / "summary.json"
    if not summary.is_file():
        raise RuntimeError("Quickstart did not create summary.json")
    print(f"Quickstart complete: {summary}")


if __name__ == "__main__":
    main()
