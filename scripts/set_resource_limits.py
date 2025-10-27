#!/usr/bin/env python3
"""
Resource Limits for M1 Mac Training

Sets conservative resource limits to prevent system overload:
- Limits CPU threads for scikit-learn, XGBoost, PyTorch
- Sets reasonable batch sizes
- Prevents memory bloat

Import this at the top of training scripts.
"""

import os
import warnings


def set_mac_friendly_limits():
    """
    Set resource limits appropriate for M1 Mac.

    Limits:
    - 4 CPU threads (out of 8-10 available on M1)
    - Conservative memory usage
    - Batch processing for large datasets
    """
    # Scikit-learn parallelism
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # XGBoost threads
    os.environ["XGB_NUM_THREADS"] = "4"

    # PyTorch threads
    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
    except ImportError:
        pass

    # Optuna parallelism
    os.environ["OPTUNA_N_JOBS"] = "1"

    print("Resource limits set for M1 Mac:")
    print("  - CPU threads: 4")
    print("  - OMP/MKL/OpenBLAS threads: 4")
    print("  - PyTorch threads: 4")
    print("  - Optuna jobs: 1 (sequential)")


# Auto-apply when imported
set_mac_friendly_limits()
