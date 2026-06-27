#!/usr/bin/env python3
"""Run a dependency and import sanity check for the CLAQ package."""

from __future__ import annotations

import platform

import numpy as np
import sklearn
import torch

import claq


def main() -> None:
    print(f"CLAQ: {claq.__version__}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        "MPS available: "
        f"{getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}"
    )
    print("Installation check passed.")


if __name__ == "__main__":
    main()
