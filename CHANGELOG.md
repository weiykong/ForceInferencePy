# Changelog

All notable changes to ForceInferencePy are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **Napari plugin** (`force_inference._napari`) exposing the full pipeline as a
  6-tab dock widget — Segment, Topology + Curvature, Solve + Stress, Visualise,
  TimeSeries, and 2.5D / 3D — wiring every pipeline parameter to a Qt control.
  Segmentation runs in a spawned subprocess to avoid the macOS OMP/PyTorch
  segfault inside Qt threads. Launch with `python -m force_inference._napari`.
- **Batchelor stress crosses** rendered as napari `Vectors` layers (principal
  axes of the per-cell stress tensor: red = tension, blue = compression).
- **Portfolio poster** generator (`scripts/generate_portfolio_poster.py`) that
  renders a recruiter-ready figure from a live pipeline run.

### Fixed

- `solve_laplace` is now called with its real signature in the napari widget
  (`regularization`, `tension_val`, `detrend`, `zero_center`, `border_margin`);
  curvature is auto-computed when the Young-Laplace solver is selected.

### Performance

- **Label-driven topology extraction is 14–21× faster** with byte-for-byte
  identical output (equivalence-tested). Two hot paths each ran a full-image
  operation inside a per-item loop (`O(n_items × H × W)`):
  - `_build_edges_from_corners` (`topology_label.py`) — replaced full-map
    connected-component labelling per cell-pair with bounding-box-local
    labelling indexed in a single pass over the corner map.
  - `_cluster_vertex_corners` (`topology_label.py`) — replaced a full
    `comp_label == cid` scan per vertex component with one-pass grouping of
    component pixels.
  - Benchmark harness and baseline/after numbers added under `benchmarks/`.

---

## [0.1.0] — 2024 (Initial Release)

### Added

**Core data structures** (`force_inference/core.py`)
- `Tissue` dataclass — unified container for 2D/2.5D tissue topology (vertices, edges, cell neighbours, label mask).
- `ForceResult` dataclass — stores inferred tensions, pressures, residual, and optional stress tensors.

**Segmentation** (`force_inference/segmentation.py`)
- `segment_grayscale` — h-minima watershed pipeline for membrane-labelled images (TIFF, PNG, JPEG).

**Topology extraction** (`force_inference/topology.py`, `force_inference/topology_label.py`)
- Skeleton-based topology extraction (`extract_topology`).
- Label-driven topology extraction (`extract_topology_label`) — handles twin junctions without skeleton merging artefacts; supports stub collapse, tiny-twin promotion, spline resampling, and sub-pixel vertex snapping.

**Geometry** (`force_inference/geometry.py`)
- `compute_curvature` — pixel tracing + circle fitting + analytical tangent computation.
- `map_z_to_vertices` — maps brightest-Z position onto tissue vertices for 2.5D stacks.
- `calculate_batchelor_stress` — per-cell 2×2 stress tensor via Batchelor formula.
- `interpolate_stress_to_grid` — Gaussian-weighted coarse-graining of cell stress onto a regular grid.

**Solvers** (`force_inference/solvers.py`)
- `solve_bayesian` — Bayesian force inference with automatic μ selection via log-evidence maximisation.
- `solve_laplace` — Laplace-pressure solver with border-cell atmosphere treatment.
- `BayesianScanResult` — structured result for μ-scan output.

**Visualization** (`force_inference/visualization.py`)
- Tension overlay, pressure map, stress ellipse, and topology diagnostic plots.

**Examples** (`examples/`)
- `demo_2d_bayesian.py`, `demo_laplace.py`, `demo_stress_analysis.py`, `demo_25d_stack.py`.
- Diagnostic scripts: `diagnose_junctions.py`, `diagnose_tif_vs_jpg.py`, `compare_methods.py`.

**Documentation**
- `README.md`, `QUICK_START.md`, `LABEL_DRIVEN_TOPOLOGY_README.md`.
