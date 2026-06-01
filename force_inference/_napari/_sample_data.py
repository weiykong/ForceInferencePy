"""Sample data provider for the napari plugin."""
from __future__ import annotations
from pathlib import Path

import numpy as np


def load_test_tissue():
    """Load the bundled test.tif as a napari sample data layer."""
    try:
        from skimage import io
    except ImportError:
        return []

    tif = Path(__file__).resolve().parents[2] / "data" / "test.tif"
    if not tif.exists():
        return []

    img = io.imread(str(tif))
    if img.ndim == 3 and img.shape[-1] <= 4:
        gray = img[..., 0].astype(float)
    else:
        gray = img.astype(float)

    lo, hi = np.percentile(gray, [0.5, 99.5])
    gray = np.clip((gray - lo) / (hi - lo + 1e-9), 0, 1)

    return [(gray, {"name": "test tissue (FI sample)", "colormap": "gray"})]
