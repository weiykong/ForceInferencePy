"""
Subprocess-safe wrappers for libraries that conflict with Qt's OMP runtime.

These functions are module-level (picklable) so they can be shipped to a
spawn-isolated subprocess via concurrent.futures.ProcessPoolExecutor.

Why subprocess?
  Cellpose v4 + PyTorch uses Intel OpenMP (libiomp5).  Qt ships its own OMP
  runtime.  On macOS, having two OMP runtimes in the same process causes a
  segfault.  Running Cellpose in a spawned subprocess gives it a clean address
  space with no Qt OMP pre-initialized.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


# ── These must be top-level (not nested) so pickle can locate them ────────────

def _run_segment_grayscale(
    img_path: str,
    h_depth: float,
    blur_sigma: float,
    min_cell_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from force_inference.segmentation import segment_grayscale
    return segment_grayscale(
        img_path,
        h_depth=h_depth,
        blur_sigma=blur_sigma,
        min_cell_size=min_cell_size,
    )


def _run_segment_cellpose(
    img_path: str,
    model_type: str,
    diameter,
    flow_threshold: float,
    cellprob_threshold: float,
    min_size: int,
    gpu: bool,
    invert: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    # Set OMP vars before torch is imported inside the subprocess
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "KMP_DUPLICATE_LIB_OK"):
        os.environ[var] = "1" if var != "KMP_DUPLICATE_LIB_OK" else "TRUE"

    from force_inference.segmentation import segment_cellpose
    return segment_cellpose(
        img_path,
        model_type=model_type,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        gpu=gpu,
        invert=invert,
    )


# ── Executor helper ───────────────────────────────────────────────────────────

def run_in_subprocess(fn, *args, timeout: int = 600, **kwargs):
    """
    Run *fn* (*args, **kwargs) in a spawned subprocess and return the result.

    Uses a single-worker ProcessPoolExecutor with the 'spawn' start method so
    the child has a clean address space (no Qt, no conflicting OMP runtime).
    """
    import concurrent.futures
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=ctx
    ) as exe:
        future = exe.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)
