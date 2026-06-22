"""ForceInferencePy — Bayesian cell force inference from fluorescence microscopy.

Quick start::

    import force_inference as fi
    labels, gray = fi.segment_grayscale("image.tif")
    tissue  = fi.extract_topology_label(labels)
    result  = fi.solve_bayesian(tissue).best_result
    print(result.summary())

Napari GUI::

    python -m force_inference._napari --image image.tif
"""

__version__ = "0.2.0"
__author__  = "weiykong"
__email__   = "weiyuankong@gmail.com"
__license__ = "MIT"

from .core import Tissue, ForceResult
from .topology import extract_topology
from .topology_label import extract_topology_label
from .solvers import solve_bayesian, solve_bayesian_3d, solve_laplace, BayesianScanResult
from .geometry import (
    map_z_to_vertices,
    calculate_batchelor_stress,
    interpolate_stress_to_grid,
    compute_curvature,
)
from .segmentation import segment_grayscale, segment_cellpose
from .timeseries import TimeSeries, align_timeseries
from .visualization import (
    plot_tensions,
    plot_pressures,
    plot_curvature,
    plot_topology_check,
    plot_stress_crosses,
    plot_cell_stress_crosses,
)

__all__ = [
    # version
    "__version__",
    # data structures
    "Tissue",
    "ForceResult",
    "BayesianScanResult",
    # segmentation
    "segment_grayscale",
    "segment_cellpose",
    # topology
    "extract_topology",
    "extract_topology_label",
    # solvers
    "solve_bayesian",
    "solve_bayesian_3d",
    "solve_laplace",
    # geometry
    "map_z_to_vertices",
    "calculate_batchelor_stress",
    "interpolate_stress_to_grid",
    "compute_curvature",
    # time-series
    "TimeSeries",
    "align_timeseries",
    # visualisation
    "plot_tensions",
    "plot_pressures",
    "plot_curvature",
    "plot_topology_check",
    "plot_stress_crosses",
    "plot_cell_stress_crosses",
]
