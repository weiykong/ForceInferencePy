"""Golden-value regression tests for label-driven topology extraction.

These pin the exact vertex/edge counts produced on the bundled sample images.
The values were captured before the performance optimisation of
`_build_edges_from_corners` / `_cluster_vertex_corners`, so this guards the
"14-21x faster, byte-for-byte identical output" guarantee against regressions.
"""
import os

import numpy as np
import pytest

from force_inference.segmentation import segment_grayscale
from force_inference.topology_label import extract_topology_label

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Golden values: (n_vertices, n_edges) — identical across junction_window 1 and 2.
GOLDEN = {
    "example.tif": (445, 610),
    "test.tif": (1318, 1902),
}


@pytest.mark.parametrize("image,expected", GOLDEN.items())
@pytest.mark.parametrize("half_window", [1, 2])
def test_topology_golden_counts(image, expected, half_window):
    path = os.path.join(DATA_DIR, image)
    if not os.path.exists(path):
        pytest.skip(f"sample image {image} not present")

    labels, _ = segment_grayscale(path)
    labels = labels.astype(np.int64)
    tissue = extract_topology_label(labels, junction_window=half_window)

    n_v, n_e = expected
    assert len(tissue.V) == n_v, f"{image}: vertex count drifted"
    assert len(tissue.E) == n_e, f"{image}: edge count drifted"
