"""ForceInferenceWidget — a Napari dock widget.

Every pipeline parameter is exposed as a typed Qt control.
"""
from __future__ import annotations

import copy
import os
import traceback
from pathlib import Path
from typing import Optional

# ── macOS OMP guard: must be set before numpy/torch are imported ──────────────
# Prevents the OMP segfault that occurs when Cellpose/PyTorch spawns OpenMP
# workers inside a Qt thread on macOS.
for _omp_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_omp_var, "1")

import numpy as np

# ── Qt imports ────────────────────────────────────────────────────────────────
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QScrollArea, QSizePolicy, QTextEdit,
    QLineEdit, QFileDialog, QStackedWidget, QFrame, QProgressBar,
)

# ── ForceInference imports ────────────────────────────────────────────────────
from ..core import Tissue, ForceResult
from ..solvers import solve_bayesian, solve_bayesian_3d, solve_laplace
from ..geometry import (
    compute_curvature, calculate_batchelor_stress,
    interpolate_stress_to_grid, map_z_to_vertices,
)
from ..topology_label import extract_topology_label
from ..split_four_way import split_high_degree_vertices
from ..timeseries import TimeSeries, align_timeseries


# ─────────────────────────────────────────────────────────────────────────────
# Small UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group(title: str, layout) -> QGroupBox:
    """Wrap a QLayout (or a QWidget) in a titled QGroupBox."""
    from qtpy.QtWidgets import QLayout
    g = QGroupBox(title)
    if isinstance(layout, QLayout):
        g.setLayout(layout)
    else:
        # layout is actually a QWidget — embed it
        box_layout = QVBoxLayout(g)
        box_layout.setContentsMargins(4, 4, 4, 4)
        box_layout.addWidget(layout)
    return g


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


def _dbl(value: float, lo: float, hi: float, step: float = 0.1,
         decimals: int = 2) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(value)
    return w


def _int(value: int, lo: int, hi: int) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    return w


def _check(value: bool, label: str = "") -> QCheckBox:
    w = QCheckBox(label)
    w.setChecked(value)
    return w


def _combo(*options: str, current: str = "") -> QComboBox:
    w = QComboBox()
    for o in options:
        w.addItem(o)
    if current:
        idx = w.findText(current)
        if idx >= 0:
            w.setCurrentIndex(idx)
    return w


def _run_btn(label: str, color: str = "#2563eb") -> QPushButton:
    b = QPushButton(label)
    b.setStyleSheet(
        f"QPushButton {{ background:{color}; color:white; font-weight:bold; "
        f"border-radius:4px; padding:6px 14px; }}"
        f"QPushButton:hover {{ background:#1d4ed8; }}"
        f"QPushButton:disabled {{ background:#555; color:#999; }}"
    )
    return b


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#a0aec0; font-weight:bold; font-size:11px;")
    return lbl


def _scroll_wrap(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setWidget(widget)
    return sa


# ─────────────────────────────────────────────────────────────────────────────
# Threading — use napari's thread_worker (handles macOS OMP/PyTorch safely)
# ─────────────────────────────────────────────────────────────────────────────

def _make_worker(fn, *args, **kwargs):
    """Wrap *fn* in a napari thread_worker and return the worker object."""
    from napari.qt.threading import thread_worker

    @thread_worker
    def _run():
        return fn(*args, **kwargs)

    return _run()   # returns a GeneratorWorker / FunctionWorker


# ─────────────────────────────────────────────────────────────────────────────
# Layer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _turbo(values: np.ndarray) -> np.ndarray:
    """Map finite float values to RGBA via turbo colormap."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("turbo")
    finite = np.isfinite(values)
    vmin = values[finite].min() if finite.any() else 0.0
    vmax = values[finite].max() if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = np.zeros((len(values), 4))
    rgba[~finite] = [0.4, 0.4, 0.4, 0.5]
    rgba[finite]  = cm(norm(values[finite]))
    return rgba


def _edges_to_shapes(tissue: Tissue):
    """Return (shapes, shape_types) for a napari Shapes layer.

    napari's "line" shape type requires exactly 2 vertices.  Curved edges
    traced from pixel paths have more; those use "path" instead.
    """
    from force_inference.visualization import _fix_edge_pixels
    shapes, stypes = [], []
    for i, (v1, v2) in enumerate(tissue.E):
        p1 = tissue.V[v1, :2][::-1]   # napari: (row, col)
        p2 = tissue.V[v2, :2][::-1]
        px = tissue.E_pixels[i] if tissue.E_pixels is not None else None
        if px is not None and len(px) > 1:
            try:
                pts = _fix_edge_pixels(px, v1_pos=tissue.V[v1, :2],
                                        v2_pos=tissue.V[v2, :2])
                coords = pts[:, ::-1]   # (row, col)
            except Exception:
                coords = np.array([p1, p2])
        else:
            coords = np.array([p1, p2])
        shapes.append(coords)
        # "line" requires exactly 2 pts; anything longer is a "path"
        stypes.append("line" if len(coords) == 2 else "path")
    return shapes, stypes


# ─────────────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────────────

class ForceInferenceWidget(QWidget):
    """Napari dock widget — exposes every ForceInferencePy parameter."""

    # Internal names for managed napari layers
    _LAYER_IMAGE      = "FI: gray image"
    _LAYER_LABELS     = "FI: segmentation"
    _LAYER_TOPOLOGY   = "FI: topology"
    _LAYER_TENSIONS   = "FI: tensions"
    _LAYER_PRESSURES  = "FI: cell pressures"
    _LAYER_STRESS_T   = "FI: stress — tension axes  (σ > 0)"
    _LAYER_STRESS_C   = "FI: stress — compression axes  (σ < 0)"
    _LAYER_STRESS_GRID= "FI: stress grid (interpolated)"

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer  = napari_viewer

        # Pipeline state
        self._img_path : Optional[Path]       = None
        self._labels   : Optional[np.ndarray] = None
        self._gray     : Optional[np.ndarray] = None
        self._tissue   : Optional[Tissue]     = None
        self._result   : Optional[ForceResult]= None
        self._timeseries = TimeSeries()

        # Active napari worker (FunctionWorker)
        self._active_worker = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Title
        title = QLabel("Cell Force Inference")
        title.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#60a5fa; padding:4px 0;"
        )
        root.addWidget(title)

        # Tab widget
        self._tabs = QTabWidget()
        self._tabs.addTab(_scroll_wrap(self._tab_segment()),    "① Segment")
        self._tabs.addTab(_scroll_wrap(self._tab_topology()),   "② Topology")
        self._tabs.addTab(_scroll_wrap(self._tab_solve()),      "③ Solve")
        self._tabs.addTab(_scroll_wrap(self._tab_geometry()),   "④ Geometry")
        self._tabs.addTab(_scroll_wrap(self._tab_visualise()),  "⑤ Visualise")
        self._tabs.addTab(_scroll_wrap(self._tab_timeseries()), "⑥ TimeSeries")
        self._tabs.addTab(_scroll_wrap(self._tab_3d()),         "⑦ 2.5D / 3D")
        root.addWidget(self._tabs)

        # Run-all button
        self._run_all_btn = _run_btn("▶  Run Full Pipeline", "#16a34a")
        self._run_all_btn.clicked.connect(self._run_full_pipeline)
        root.addWidget(self._run_all_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)       # indeterminate
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        root.addWidget(self._progress)

        # Status log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(90)
        self._log.setStyleSheet(
            "background:#111; color:#a0aec0; font-size:10px;"
            "font-family:monospace; border-radius:4px;"
        )
        root.addWidget(self._log)

    # ── Tab 1: Segmentation ───────────────────────────────────────────────────

    def _tab_segment(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Image source
        src_form = QFormLayout()
        self._img_edit = QLineEdit("(no file selected)")
        self._img_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_image)
        row = QHBoxLayout()
        row.addWidget(self._img_edit, 1)
        row.addWidget(browse_btn)
        src_form.addRow("Image file:", row)
        v.addLayout(src_form)

        # Method selector
        self._seg_method = _combo("Grayscale (watershed)", "Cellpose",
                                   current="Grayscale (watershed)")
        self._seg_method.currentIndexChanged.connect(self._on_seg_method_changed)
        mf = QFormLayout(); mf.addRow("Method:", self._seg_method)
        v.addLayout(mf)

        # Stack for method-specific params
        self._seg_stack = QStackedWidget()

        # ── Grayscale params ──
        gs_widget = QWidget(); gs_form = QFormLayout(gs_widget)
        self._gs_h_depth    = _dbl(2.0, 0.1, 100.0, 0.5, 2)
        self._gs_blur_sigma = _dbl(1.0, 0.0, 20.0, 0.5, 2)
        self._gs_min_cell   = _int(20, 1, 10000)
        gs_form.addRow("h_depth:", self._gs_h_depth)
        gs_form.addRow("blur_sigma:", self._gs_blur_sigma)
        gs_form.addRow("min_cell_size:", self._gs_min_cell)
        self._seg_stack.addWidget(gs_widget)

        # ── Cellpose params ──
        cp_widget = QWidget(); cp_form = QFormLayout(cp_widget)
        self._cp_model      = _combo("cyto3", "cpsam", "cyto2", "nuclei",
                                      current="cyto3")
        self._cp_diameter   = _dbl(0.0, 0.0, 2000.0, 5.0, 1)
        self._cp_diameter.setSpecialValueText("auto")
        self._cp_flow_thr   = _dbl(0.4, 0.0, 5.0, 0.05, 3)
        self._cp_prob_thr   = _dbl(0.0, -8.0, 8.0, 0.5, 2)
        self._cp_min_size   = _int(15, 1, 10000)
        self._cp_gpu        = _check(True, "Use GPU")
        self._cp_invert     = _check(False, "Invert image")
        cp_form.addRow("model_type:", self._cp_model)
        cp_form.addRow("diameter (px):", self._cp_diameter)
        cp_form.addRow("flow_threshold:", self._cp_flow_thr)
        cp_form.addRow("cellprob_threshold:", self._cp_prob_thr)
        cp_form.addRow("min_size (px):", self._cp_min_size)
        cp_form.addRow("", self._cp_gpu)
        cp_form.addRow("", self._cp_invert)
        self._seg_stack.addWidget(cp_widget)

        # Wrap the QStackedWidget in a QGroupBox manually (setLayout needs a QLayout)
        params_box = QGroupBox("Parameters")
        params_box_layout = QVBoxLayout(params_box)
        params_box_layout.setContentsMargins(4, 4, 4, 4)
        params_box_layout.addWidget(self._seg_stack)
        v.addWidget(params_box)
        v.addWidget(_hline())

        self._seg_btn = _run_btn("▶  Segment")
        self._seg_btn.clicked.connect(self._run_segment)
        v.addWidget(self._seg_btn)
        v.addStretch()
        return w

    # ── Tab 2: Topology ───────────────────────────────────────────────────────

    def _tab_topology(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # ── extract_topology_label — ALL real parameters ──────────────────────
        ext_form = QFormLayout()
        self._topo_min_edge_len       = _int(2,    0, 500)
        self._topo_trace_pixels       = _check(True,  "trace_pixels")
        self._topo_use_skel           = _check(True,  "use_skeleton_geometry")
        self._topo_clean              = _check(False, "clean (remove stray pixels)")
        self._topo_min_clean_edge     = _dbl(3.0, 0.1, 100.0, 0.5, 1)
        self._topo_remove_outer       = _check(False, "remove_outer_layer")
        self._topo_vertex_cluster_r   = _dbl(2.0, 0.0,  50.0, 0.5, 1)
        self._topo_curve_points       = _int(0,    0, 100)
        self._topo_collapse_stubs     = _check(True,  "collapse_stubs")
        self._topo_stub_threshold     = _dbl(3.0, 0.1, 100.0, 0.5, 1)
        self._topo_collapse_twins     = _check(False, "collapse_tiny_twins")
        self._topo_twin_threshold     = _dbl(3.0, 0.1, 100.0, 0.5, 1)
        self._topo_junction_window    = _int(0,    0,  50)
        self._topo_dilate_labels      = _int(0,    0,  20)

        ext_form.addRow("min_edge_len:",      self._topo_min_edge_len)
        ext_form.addRow("",                   self._topo_trace_pixels)
        ext_form.addRow("",                   self._topo_use_skel)
        ext_form.addRow("",                   self._topo_clean)
        ext_form.addRow("min_clean_edge_len:",self._topo_min_clean_edge)
        ext_form.addRow("",                   self._topo_remove_outer)
        ext_form.addRow("vertex_cluster_r:",  self._topo_vertex_cluster_r)
        ext_form.addRow("curve_points:",      self._topo_curve_points)
        ext_form.addRow("",                   self._topo_collapse_stubs)
        ext_form.addRow("stub_edge_threshold:",self._topo_stub_threshold)
        ext_form.addRow("",                   self._topo_collapse_twins)
        ext_form.addRow("tiny_twin_threshold:",self._topo_twin_threshold)
        ext_form.addRow("junction_window:",   self._topo_junction_window)
        ext_form.addRow("dilate_labels:",     self._topo_dilate_labels)
        v.addWidget(_group("extract_topology_label", ext_form))

        # split_high_degree_vertices params
        split_form = QFormLayout()
        self._split_enable      = _check(True, "Enable 4-way→3-way splitting")
        self._split_length      = _dbl(4.0, 0.5, 100.0, 0.5, 2)
        split_form.addRow("", self._split_enable)
        split_form.addRow("split_length:", self._split_length)
        self._split_enable.stateChanged.connect(
            lambda s: self._split_length.setEnabled(bool(s)))
        v.addWidget(_group("split_high_degree_vertices", split_form))

        v.addWidget(_hline())
        self._topo_btn = _run_btn("▶  Extract Topology")
        self._topo_btn.clicked.connect(self._run_topology)
        v.addWidget(self._topo_btn)
        v.addStretch()
        return w

    # ── Tab 3: Solve ──────────────────────────────────────────────────────────

    def _tab_solve(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Solver selection
        solver_form = QFormLayout()
        self._solver_type = _combo(
            "Bayesian 2D",
            "Bayesian 3D (Newell)",
            "Young-Laplace",
        )
        self._solver_type.currentIndexChanged.connect(
            self._on_solver_changed)
        solver_form.addRow("Solver:", self._solver_type)
        v.addLayout(solver_form)

        # Shared params
        shared_form = QFormLayout()
        self._solve_excl_border  = _check(True, "Exclude border edges")
        self._solve_border_margin = _int(5, 0, 200)
        shared_form.addRow("", self._solve_excl_border)
        shared_form.addRow("border_margin:", self._solve_border_margin)
        v.addWidget(_group("Shared parameters", shared_form))

        # Bayesian-only params (hidden for Laplace)
        self._bayes_group = QGroupBox("Bayesian parameters")
        bayes_form = QFormLayout(self._bayes_group)
        self._mu_auto   = _check(True, "Auto-select μ (evidence maximisation)")
        self._mu_value  = _dbl(1e-3, 1e-12, 1e3, 1e-3, 8)
        self._mu_value.setEnabled(False)
        self._mu_auto.stateChanged.connect(
            lambda s: self._mu_value.setEnabled(not bool(s)))
        bayes_form.addRow("", self._mu_auto)
        bayes_form.addRow("μ (manual):", self._mu_value)
        v.addWidget(self._bayes_group)

        v.addWidget(_hline())
        self._solve_btn = _run_btn("▶  Solve", "#7c3aed")
        self._solve_btn.clicked.connect(self._run_solve)
        v.addWidget(self._solve_btn)
        v.addStretch()
        return w

    # ── Tab 4: Geometry ───────────────────────────────────────────────────────

    def _tab_geometry(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Curvature
        curv_form = QFormLayout()
        self._do_curvature = _check(True, "Compute curvature after topology")
        curv_form.addRow("", self._do_curvature)
        v.addWidget(_group("compute_curvature", curv_form))

        # Batchelor stress
        bstress_form = QFormLayout()
        self._do_batchelor = _check(True, "Calculate Batchelor stress tensors")
        bstress_form.addRow("", self._do_batchelor)
        v.addWidget(_group("calculate_batchelor_stress", bstress_form))

        # Grid interpolation
        grid_form = QFormLayout()
        self._do_grid_interp     = _check(False, "Interpolate stress to grid")
        self._grid_size          = _int(50, 5, 500)
        self._grid_sigma         = _dbl(0.0, 0.0, 500.0, 5.0, 1)
        self._grid_sigma.setSpecialValueText("auto (1.5 × grid_size)")
        self._do_grid_interp.stateChanged.connect(
            lambda s: [self._grid_size.setEnabled(bool(s)),
                       self._grid_sigma.setEnabled(bool(s))])
        self._grid_size.setEnabled(False)
        self._grid_sigma.setEnabled(False)
        grid_form.addRow("", self._do_grid_interp)
        grid_form.addRow("grid_size:", self._grid_size)
        grid_form.addRow("smoothing_sigma:", self._grid_sigma)
        v.addWidget(_group("interpolate_stress_to_grid", grid_form))

        v.addWidget(_hline())
        self._geom_btn = _run_btn("▶  Run Geometry", "#ea580c")
        self._geom_btn.clicked.connect(self._run_geometry)
        v.addWidget(self._geom_btn)
        v.addStretch()
        return w

    # ── Tab 5: Visualise ──────────────────────────────────────────────────────

    def _tab_visualise(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # Topology layer
        topo_vis_form = QFormLayout()
        self._vis_topo       = _check(True, "Show topology edges")
        self._vis_topo_width = _dbl(1.5, 0.3, 10.0, 0.5, 1)
        topo_vis_form.addRow("", self._vis_topo)
        topo_vis_form.addRow("Edge width:", self._vis_topo_width)
        v.addWidget(_group("Topology layer", topo_vis_form))

        # Tension layer
        tens_form = QFormLayout()
        self._vis_tensions    = _check(True, "Show tensions (colour map)")
        self._vis_tens_cmap   = _combo("turbo", "plasma", "viridis",
                                        "magma", "RdBu_r", "seismic")
        self._vis_tens_width  = _dbl(2.0, 0.3, 10.0, 0.5, 1)
        self._vis_tens_vmin   = _dbl(0.0, 0.0, 1000.0, 0.01, 4)
        self._vis_tens_vmin.setSpecialValueText("auto (2nd percentile)")
        self._vis_tens_vmax   = _dbl(0.0, 0.0, 1000.0, 0.01, 4)
        self._vis_tens_vmax.setSpecialValueText("auto (98th percentile)")
        tens_form.addRow("", self._vis_tensions)
        tens_form.addRow("Colormap:", self._vis_tens_cmap)
        tens_form.addRow("Edge width:", self._vis_tens_width)
        tens_form.addRow("vmin:", self._vis_tens_vmin)
        tens_form.addRow("vmax:", self._vis_tens_vmax)
        v.addWidget(_group("Tension layer", tens_form))

        # ── Cell pressures layer ──────────────────────────────────────────────
        pres_form = QFormLayout()
        self._vis_pressures     = _check(True, "Show cell pressures")
        self._vis_pres_cmap     = _combo("RdYlBu_r", "coolwarm", "seismic",
                                          "PuOr_r", "RdBu_r")
        self._vis_pres_opacity  = _dbl(0.55, 0.0, 1.0, 0.05, 2)
        pres_form.addRow("", self._vis_pressures)
        pres_form.addRow("Colormap:", self._vis_pres_cmap)
        pres_form.addRow("Opacity:", self._vis_pres_opacity)
        v.addWidget(_group("Cell pressures layer", pres_form))

        # ── Batchelor stress — principal-axis crosses ─────────────────────────
        #
        # Each cell gets two symmetric arrow pairs from the eigenvectors of its
        # 2×2 stress tensor:
        #   red  arms  (tension layer)    → positive eigenvalue
        #   blue arms  (compression layer) → negative eigenvalue
        #   length = |eigenvalue| × scale
        #
        bstress_vis_form = QFormLayout()
        self._vis_batchelor       = _check(True,  "Show principal-stress crosses")
        self._vis_batch_scale     = _dbl(20.0, 0.1, 500.0, 5.0, 1)
        self._vis_batch_min_mag   = _dbl(0.0,  0.0, 1e6,   0.001, 4)
        self._vis_batch_min_mag.setSpecialValueText("no threshold")
        self._vis_batch_tension_col   = QLineEdit("#ff3333")   # editable hex
        self._vis_batch_compress_col  = QLineEdit("#3377ff")
        bstress_vis_form.addRow("", self._vis_batchelor)
        bstress_vis_form.addRow("Arm scale:", self._vis_batch_scale)
        bstress_vis_form.addRow("Min |eigenvalue|:", self._vis_batch_min_mag)
        bstress_vis_form.addRow("Tension colour:", self._vis_batch_tension_col)
        bstress_vis_form.addRow("Compression colour:", self._vis_batch_compress_col)
        v.addWidget(_group(
            "Batchelor stress — principal axes  (eigenvectors of σ)",
            bstress_vis_form))

        # ── Interpolated stress grid ──────────────────────────────────────────
        sgrid_form = QFormLayout()
        self._vis_stress_grid   = _check(False, "Show interpolated stress grid")
        sgrid_form.addRow("", self._vis_stress_grid)
        v.addWidget(_group("Interpolated grid layer", sgrid_form))

        # ── Stress principal-axis crosses ─────────────────────────────────────
        stress_vis_form = QFormLayout()
        self._vis_stress       = _check(False, "Show stress crosses (matplotlib)")
        self._vis_stress_scale = _dbl(60.0, 1.0, 500.0, 5.0, 1)
        self._vis_stress_min   = _dbl(0.01, 0.0, 100.0, 0.01, 4)
        stress_vis_form.addRow("", self._vis_stress)
        stress_vis_form.addRow("Scale:", self._vis_stress_scale)
        stress_vis_form.addRow("min_mag:", self._vis_stress_min)
        v.addWidget(_group("Stress principal-axis crosses", stress_vis_form))

        v.addWidget(_hline())
        self._vis_btn = _run_btn("▶  Update Layers", "#0891b2")
        self._vis_btn.clicked.connect(self._run_visualise)
        v.addWidget(self._vis_btn)
        v.addStretch()
        return w

    # ── Tab 6: Time-series ────────────────────────────────────────────────────

    def _tab_timeseries(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        info = QLabel(
            "Add the current solved frame to a TimeSeries, then align.\n"
            "Run Segment + Topology + Solve for each time point, then\n"
            "click 'Add Frame' before moving to the next."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#94a3b8; font-size:10px;")
        v.addWidget(info)

        ts_frame_form = QFormLayout()
        self._ts_time_val = _dbl(0.0, -1e9, 1e9, 1.0, 2)
        ts_frame_form.addRow("Frame time:", self._ts_time_val)
        v.addWidget(_group("Current frame", ts_frame_form))

        ts_align_form = QFormLayout()
        self._ts_strategy   = _combo("shared_edges", "median",
                                      "fluorescence", "fixed_edge")
        self._ts_ref_frame  = _int(0, 0, 9999)
        ts_align_form.addRow("strategy:", self._ts_strategy)
        ts_align_form.addRow("reference_frame:", self._ts_ref_frame)
        v.addWidget(_group("align_timeseries parameters", ts_align_form))

        # Status
        self._ts_status_lbl = QLabel("TimeSeries: 0 frames")
        self._ts_status_lbl.setStyleSheet("color:#60a5fa; font-size:11px;")
        v.addWidget(self._ts_status_lbl)

        btn_row = QHBoxLayout()
        self._ts_add_btn    = _run_btn("+ Add Frame",  "#0f766e")
        self._ts_align_btn  = _run_btn("⇄ Align",     "#7c3aed")
        self._ts_clear_btn  = _run_btn("✕ Clear",     "#dc2626")
        self._ts_add_btn.clicked.connect(self._ts_add_frame)
        self._ts_align_btn.clicked.connect(self._ts_align)
        self._ts_clear_btn.clicked.connect(self._ts_clear)
        btn_row.addWidget(self._ts_add_btn)
        btn_row.addWidget(self._ts_align_btn)
        btn_row.addWidget(self._ts_clear_btn)
        v.addLayout(btn_row)
        v.addStretch()
        return w

    # ── Tab 7: 2.5D / 3D ─────────────────────────────────────────────────────

    def _tab_3d(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        info = QLabel(
            "Map Z coordinates from a confocal Z-stack onto tissue vertices,\n"
            "then use the 3D Bayesian solver for full out-of-plane force balance."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#94a3b8; font-size:10px;")
        v.addWidget(info)

        z_form = QFormLayout()
        self._z_stack_name  = QLineEdit("(enter napari layer name)")
        self._z_xy_radius   = _int(1, 0, 20)
        self._z_fallback    = _dbl(0.0, -1e4, 1e4, 1.0, 2)
        z_form.addRow("Z-stack layer name:", self._z_stack_name)
        z_form.addRow("xy_radius:", self._z_xy_radius)
        z_form.addRow("z_fallback:", self._z_fallback)
        v.addWidget(_group("map_z_to_vertices", z_form))

        self._z_map_btn = _run_btn("▶  Map Z to Vertices", "#1d4ed8")
        self._z_map_btn.clicked.connect(self._run_z_map)
        v.addWidget(self._z_map_btn)

        v.addWidget(_hline())
        info3d = QLabel(
            "After mapping Z, switch the solver in ③ to 'Bayesian 3D' and\n"
            "click '▶ Solve'. The Newell-normal pressure term is used automatically."
        )
        info3d.setWordWrap(True)
        info3d.setStyleSheet("color:#94a3b8; font-size:10px;")
        v.addWidget(info3d)
        v.addStretch()
        return w

    # ── Signal handlers / private callbacks ───────────────────────────────────

    def _on_seg_method_changed(self, idx: int):
        self._seg_stack.setCurrentIndex(idx)

    def _on_solver_changed(self, idx: int):
        is_bayes = idx in (0, 1)
        self._bayes_group.setVisible(is_bayes)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "", "Images (*.tif *.tiff *.png *.jpg *.npy)"
        )
        if path:
            self._img_path = Path(path)
            self._img_edit.setText(str(self._img_path))

    # ── Pipeline execution ────────────────────────────────────────────────────

    def _log_msg(self, msg: str):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())

    def _set_busy(self, busy: bool):
        self._progress.setVisible(busy)
        for btn in (self._seg_btn, self._topo_btn, self._solve_btn,
                    self._geom_btn, self._vis_btn, self._run_all_btn,
                    self._z_map_btn):
            btn.setEnabled(not busy)

    def _start_worker(self, fn, *args, on_done=None, label="Working…", **kwargs):
        if self._active_worker is not None:
            self._log_msg("⚠ Already running — please wait.")
            return
        self._set_busy(True)
        self._log_msg(f"► {label}")

        worker = _make_worker(fn, *args, **kwargs)
        self._active_worker = worker

        def _on_returned(result):
            self._active_worker = None
            self._set_busy(False)
            if on_done:
                try:
                    on_done(result)
                except Exception:
                    self._log_msg(f"✗ Display error:\n{traceback.format_exc()}")

        def _on_error(exc):
            self._active_worker = None
            self._set_busy(False)
            self._log_msg(f"✗ Error: {exc}\n{traceback.format_exc()}")

        worker.returned.connect(_on_returned)
        worker.errored.connect(_on_error)
        worker.start()

    # ── Segment ───────────────────────────────────────────────────────────────

    def _run_segment(self):
        if self._img_path is None or not self._img_path.exists():
            self._log_msg("✗ No valid image file selected.")
            return

        method = self._seg_method.currentIndex()
        if method == 0:
            fn = self._do_segment_gray
        else:
            fn = self._do_segment_cellpose
        self._start_worker(fn, label="Segmenting…", on_done=self._after_segment)

    def _do_segment_gray(self):
        # Grayscale uses skimage only — still run in subprocess for consistency
        from ._subprocess_helpers import _run_segment_grayscale, run_in_subprocess
        return run_in_subprocess(
            _run_segment_grayscale,
            str(self._img_path),
            self._gs_h_depth.value(),
            self._gs_blur_sigma.value(),
            self._gs_min_cell.value(),
        )

    def _do_segment_cellpose(self):
        # MUST run in a spawn subprocess — Cellpose/PyTorch OMP segfaults inside Qt
        from ._subprocess_helpers import _run_segment_cellpose, run_in_subprocess
        diam = self._cp_diameter.value()
        return run_in_subprocess(
            _run_segment_cellpose,
            str(self._img_path),
            self._cp_model.currentText(),
            None if diam == 0.0 else diam,
            self._cp_flow_thr.value(),
            self._cp_prob_thr.value(),
            self._cp_min_size.value(),
            self._cp_gpu.isChecked(),
            self._cp_invert.isChecked(),
        )

    def _after_segment(self, result):
        labels, gray = result
        self._labels = labels
        self._gray   = gray
        n = int(labels.max())
        self._log_msg(f"✓ Segmentation done — {n} cells.")
        # Push to napari
        self._upsert_layer("image",  self._LAYER_IMAGE,  gray,   {"colormap": "gray"})
        self._upsert_layer("labels", self._LAYER_LABELS, labels, {})

    # ── Topology ──────────────────────────────────────────────────────────────

    def _run_topology(self):
        if self._labels is None:
            self._log_msg("✗ Segment first.")
            return
        self._start_worker(self._do_topology, label="Extracting topology…",
                           on_done=self._after_topology)

    def _do_topology(self):
        tissue = extract_topology_label(
            self._labels,
            min_edge_len         = self._topo_min_edge_len.value(),
            trace_pixels         = self._topo_trace_pixels.isChecked(),
            use_skeleton_geometry= self._topo_use_skel.isChecked(),
            clean                = self._topo_clean.isChecked(),
            min_clean_edge_len   = self._topo_min_clean_edge.value(),
            remove_outer_layer   = self._topo_remove_outer.isChecked(),
            vertex_cluster_r     = self._topo_vertex_cluster_r.value(),
            curve_points         = self._topo_curve_points.value(),
            collapse_stubs       = self._topo_collapse_stubs.isChecked(),
            stub_edge_threshold  = self._topo_stub_threshold.value(),
            collapse_tiny_twins  = self._topo_collapse_twins.isChecked(),
            tiny_twin_threshold  = self._topo_twin_threshold.value(),
            junction_window      = self._topo_junction_window.value(),
            dilate_labels        = self._topo_dilate_labels.value(),
        )
        if tissue is None:
            raise RuntimeError("Topology extraction returned None.")
        if self._split_enable.isChecked():
            tissue = split_high_degree_vertices(
                copy.deepcopy(tissue),
                split_length=self._split_length.value(),
            )
        return tissue

    def _after_topology(self, tissue: Tissue):
        self._tissue = tissue
        nv, ne = len(tissue.V), len(tissue.E)
        self._log_msg(f"✓ Topology — {nv} vertices, {ne} edges.")
        self._refresh_topology_layer()

    def _refresh_topology_layer(self):
        if self._tissue is None:
            return
        if not (self._vis_topo.isChecked() if hasattr(self, "_vis_topo") else True):
            return
        shapes, stypes = _edges_to_shapes(self._tissue)
        lw = self._vis_topo_width.value() if hasattr(self, "_vis_topo_width") else 1.5
        self._upsert_layer("shapes", self._LAYER_TOPOLOGY, shapes, {
            "shape_type": stypes,
            "edge_color": "white",
            "face_color": "transparent",
            "edge_width": lw,
            "opacity": 0.8,
        })

    # ── Solve ──────────────────────────────────────────────────────────────────

    def _run_solve(self):
        if self._tissue is None:
            self._log_msg("✗ Extract topology first.")
            return
        self._start_worker(self._do_solve, label="Solving…",
                           on_done=self._after_solve)

    def _do_solve(self):
        mu    = None if self._mu_auto.isChecked() else self._mu_value.value()
        excl  = self._solve_excl_border.isChecked()
        margin= self._solve_border_margin.value()
        idx   = self._solver_type.currentIndex()
        def _unwrap(r):
            """Return a ForceResult whether r is a ForceResult or BayesianScanResult."""
            if r is None:
                return None
            return r.best_result if hasattr(r, "best_result") else r

        if idx == 0:
            return _unwrap(solve_bayesian(self._tissue, mu=mu,
                                          exclude_border_edges=excl,
                                          border_margin=margin))
        elif idx == 1:
            return _unwrap(solve_bayesian_3d(self._tissue, mu=mu,
                                             exclude_border_edges=excl,
                                             border_margin=margin))
        else:
            return solve_laplace(self._tissue,
                                  exclude_border_edges=excl,
                                  border_margin=margin)

    def _after_solve(self, result: Optional[ForceResult]):
        if result is None:
            self._log_msg("✗ Solver returned no result.")
            return
        self._result = result
        valid = int(np.sum(np.isfinite(result.tensions)))
        self._log_msg(f"✓ Solve done — {valid} tensions, "
                      f"residual={result.residual:.4g}")
        self._log_msg(result.summary())
        self._refresh_tension_layer()
        self._refresh_pressure_layer()

    def _refresh_tension_layer(self):
        if self._tissue is None or self._result is None:
            return
        shapes, stypes = _edges_to_shapes(self._tissue)
        colors = _turbo(self._result.tensions)

        # Honour manual vmin/vmax if non-zero
        finite = np.isfinite(self._result.tensions)
        vmin_ui = self._vis_tens_vmin.value() if hasattr(self, "_vis_tens_vmin") else 0.0
        vmax_ui = self._vis_tens_vmax.value() if hasattr(self, "_vis_tens_vmax") else 0.0
        if vmin_ui != 0.0 or vmax_ui != 0.0:
            import matplotlib.colors as mc
            import matplotlib.pyplot as plt
            cm = plt.get_cmap(
                self._vis_tens_cmap.currentText()
                if hasattr(self, "_vis_tens_cmap") else "turbo")
            vmin = vmin_ui if vmin_ui != 0.0 else float(np.nanpercentile(
                self._result.tensions[finite], 2))
            vmax = vmax_ui if vmax_ui != 0.0 else float(np.nanpercentile(
                self._result.tensions[finite], 98))
            norm  = mc.Normalize(vmin=vmin, vmax=vmax)
            colors = np.zeros((len(self._result.tensions), 4))
            colors[~finite] = [0.4, 0.4, 0.4, 0.4]
            colors[finite]  = cm(norm(self._result.tensions[finite]))

        lw = self._vis_tens_width.value() if hasattr(self, "_vis_tens_width") else 2.0
        self._upsert_layer("shapes", self._LAYER_TENSIONS, shapes, {
            "shape_type": stypes,
            "edge_color": colors,
            "face_color": "transparent",
            "edge_width": lw,
            "opacity": 0.95,
        })

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _run_geometry(self):
        if self._tissue is None:
            self._log_msg("✗ Extract topology first.")
            return
        self._start_worker(self._do_geometry, label="Running geometry…",
                           on_done=self._after_geometry)

    def _do_geometry(self):
        t = copy.deepcopy(self._tissue)
        r = copy.deepcopy(self._result) if self._result else None
        if self._do_curvature.isChecked():
            t = compute_curvature(t)
        if r is not None and self._do_batchelor.isChecked():
            r = calculate_batchelor_stress(t, r)
        grid = None
        if r is not None and self._do_grid_interp.isChecked():
            sigma = self._grid_sigma.value()
            grid = interpolate_stress_to_grid(
                t, r,
                grid_size     = self._grid_size.value(),
                smoothing_sigma = None if sigma == 0.0 else sigma,
            )
        return t, r, grid

    def _after_geometry(self, payload):
        tissue, result, grid = payload
        self._tissue = tissue
        self._result = result
        has_k = tissue.E_curvature is not None
        has_s = result is not None and result.stress_tensors is not None
        self._log_msg(
            f"✓ Geometry done — curvature={'yes' if has_k else 'no'}, "
            f"stress={'yes' if has_s else 'no'}."
        )
        if grid is not None:
            (gx, gy), gt = grid
            if gt is not None:
                # Trace of stress tensor = hydrostatic component
                trace = gt[..., 0, 0] + gt[..., 1, 1]
                gs = self._grid_size.value()
                # scale + translate so the grid aligns with the full image
                self._upsert_layer("image", self._LAYER_STRESS_GRID, trace, {
                    "colormap": "RdBu_r",
                    "opacity":  0.50,
                    "scale":    [gs, gs],
                    "translate":[gs / 2.0, gs / 2.0],
                })
                self._log_msg(
                    f"✓ Stress grid: {trace.shape} cells × {gs}px → "
                    f"covers ~{trace.shape[0]*gs}×{trace.shape[1]*gs}px")
        self._refresh_tension_layer()
        self._refresh_batchelor_layer()

    # ── Per-cell pressure / stress layer refresh ──────────────────────────────

    def _refresh_pressure_layer(self):
        """Render per-cell pressures as a full-resolution image layer."""
        if self._result is None or self._labels is None:
            return
        if not (self._vis_pressures.isChecked()
                if hasattr(self, "_vis_pressures") else True):
            return
        H, W = self._labels.shape
        pmap = np.full((H, W), np.nan, dtype=float)
        for ci, p in enumerate(self._result.pressures):
            mask = self._labels == (ci + 1)
            if mask.any():
                pmap[mask] = float(p)
        cmap = (self._vis_pres_cmap.currentText()
                if hasattr(self, "_vis_pres_cmap") else "RdYlBu_r")
        opacity = (self._vis_pres_opacity.value()
                   if hasattr(self, "_vis_pres_opacity") else 0.55)
        self._upsert_layer("image", self._LAYER_PRESSURES, pmap, {
            "colormap": cmap,
            "opacity":  opacity,
        })

    def _refresh_batchelor_layer(self):
        """Render per-cell Batchelor stress as principal-axis Vectors layers.

        The 2×2 stress tensor σ is diagonalised:  σ = V diag(λ₁,λ₂) Vᵀ
        Each eigenvector Vⱼ gives the direction of a principal stress axis.
        λⱼ > 0 → tension arm (red),  λⱼ < 0 → compression arm (blue).
        Each arm is drawn symmetrically (both ±Vⱼ) so it looks like a cross.

        Arm length = |λⱼ| × scale.
        """
        if self._result is None or self._result.stress_tensors is None:
            return
        if self._tissue is None:
            return
        if not (self._vis_batchelor.isChecked()
                if hasattr(self, "_vis_batchelor") else True):
            return

        scale   = (self._vis_batch_scale.value()
                   if hasattr(self, "_vis_batch_scale") else 20.0)
        min_mag = (self._vis_batch_min_mag.value()
                   if hasattr(self, "_vis_batch_min_mag") else 0.0)

        # Accumulate (pos, direction) pairs for each sign
        vecs_t: list = []   # tension  (λ > 0)
        vecs_c: list = []   # compression (λ < 0)

        centroids = self._tissue.C_centroids   # (n_cells, 2)  [x, y]
        n = min(len(self._result.stress_tensors), len(centroids))

        for ci in range(n):
            st = self._result.stress_tensors[ci, :2, :2]
            cx, cy = centroids[ci]
            pos = np.array([cy, cx], dtype=float)   # napari: (row, col)

            try:
                eigvals, eigvecs = np.linalg.eigh(st)
            except np.linalg.LinAlgError:
                continue

            for j in range(2):
                lam = float(eigvals[j])
                if abs(lam) < min_mag:
                    continue
                arm_len = abs(lam) * scale
                # eigvec is (ex, ey) in image (x,y) → convert to (drow, dcol)
                evec_rc = np.array([eigvecs[1, j], eigvecs[0, j]]) * arm_len

                # Symmetric cross: one arm each way from the centroid
                target = vecs_t if lam >= 0.0 else vecs_c
                target.append([pos,  evec_rc])
                target.append([pos, -evec_rc])

        t_col = (self._vis_batch_tension_col.text().strip()
                 if hasattr(self, "_vis_batch_tension_col") else "#ff3333")
        c_col = (self._vis_batch_compress_col.text().strip()
                 if hasattr(self, "_vis_batch_compress_col") else "#3377ff")

        for layer_name, vecs, col in [
            (self._LAYER_STRESS_T, vecs_t, t_col),
            (self._LAYER_STRESS_C, vecs_c, c_col),
        ]:
            if vecs:
                arr = np.array(vecs, dtype=float)  # (N, 2, 2)
                self._upsert_layer("vectors", layer_name, arr, {
                    "edge_color":    col,
                    "length":        1.0,
                    "vector_style":  "line",
                    "opacity":       0.85,
                })
            else:
                # No vectors of this sign — remove stale layer if present
                if layer_name in self._viewer.layers:
                    self._viewer.layers.remove(self._viewer.layers[layer_name])

    # ── Visualise (re-render) ─────────────────────────────────────────────────

    def _run_visualise(self):
        self._refresh_topology_layer()
        self._refresh_tension_layer()
        self._refresh_pressure_layer()
        self._refresh_batchelor_layer()
        self._log_msg("✓ Layers updated.")

    # ── Time-series ───────────────────────────────────────────────────────────

    def _ts_add_frame(self):
        if self._tissue is None or self._result is None:
            self._log_msg("✗ Need tissue + solve result to add a frame.")
            return
        t = self._ts_time_val.value()
        self._timeseries.add_frame(
            copy.deepcopy(self._tissue),
            copy.deepcopy(self._result),
            time=t,
        )
        self._ts_status_lbl.setText(
            f"TimeSeries: {len(self._timeseries)} frames")
        self._log_msg(
            f"✓ Added frame t={t:.2f} — total {len(self._timeseries)} frames.")
        self._ts_time_val.setValue(t + 1.0)

    def _ts_align(self):
        if len(self._timeseries) < 2:
            self._log_msg("✗ Need at least 2 frames to align.")
            return
        self._timeseries.align(
            strategy       = self._ts_strategy.currentText(),
            reference_frame= self._ts_ref_frame.value(),
        )
        scales = np.round(self._timeseries.scales, 4)
        self._log_msg(f"✓ Aligned — scale factors: {scales}")

    def _ts_clear(self):
        self._timeseries = TimeSeries()
        self._ts_status_lbl.setText("TimeSeries: 0 frames")
        self._log_msg("TimeSeries cleared.")

    # ── Z-mapping ─────────────────────────────────────────────────────────────

    def _run_z_map(self):
        if self._tissue is None:
            self._log_msg("✗ Extract topology first.")
            return
        name = self._z_stack_name.text().strip()
        if name not in self._viewer.layers:
            self._log_msg(f"✗ Layer '{name}' not found in napari.")
            return
        stack = np.asarray(self._viewer.layers[name].data)
        if stack.ndim == 3:
            tissue = map_z_to_vertices(
                copy.deepcopy(self._tissue), stack,
                xy_radius  = self._z_xy_radius.value(),
                z_fallback = self._z_fallback.value(),
            )
            self._tissue = tissue
            z_vals = tissue.V[:, 2]
            self._log_msg(
                f"✓ Z mapped — range [{z_vals.min():.1f}, {z_vals.max():.1f}]")
        else:
            self._log_msg("✗ Selected layer is not a 3D (Z, Y, X) array.")

    # ── Run full pipeline ─────────────────────────────────────────────────────

    def _run_full_pipeline(self):
        """Segment → topology → solve in sequence."""
        if self._img_path is None or not self._img_path.exists():
            self._log_msg("✗ Select an image file first (Tab ①).")
            return

        def _pipeline():
            # 1. Segment (always via subprocess — isolates OMP from Qt)
            method = self._seg_method.currentIndex()
            if method == 0:
                labels, gray = self._do_segment_gray()
            else:
                labels, gray = self._do_segment_cellpose()

            # 2. Topology
            self._labels = labels
            self._gray   = gray
            tissue = extract_topology_label(
                labels,
                min_edge_len         = self._topo_min_edge_len.value(),
                trace_pixels         = self._topo_trace_pixels.isChecked(),
                use_skeleton_geometry= self._topo_use_skel.isChecked(),
                clean                = self._topo_clean.isChecked(),
                min_clean_edge_len   = self._topo_min_clean_edge.value(),
                remove_outer_layer   = self._topo_remove_outer.isChecked(),
                vertex_cluster_r     = self._topo_vertex_cluster_r.value(),
                curve_points         = self._topo_curve_points.value(),
                collapse_stubs       = self._topo_collapse_stubs.isChecked(),
                stub_edge_threshold  = self._topo_stub_threshold.value(),
                collapse_tiny_twins  = self._topo_collapse_twins.isChecked(),
                tiny_twin_threshold  = self._topo_twin_threshold.value(),
                junction_window      = self._topo_junction_window.value(),
                dilate_labels        = self._topo_dilate_labels.value(),
            )
            if tissue is None:
                raise RuntimeError("Topology extraction returned None.")
            if self._split_enable.isChecked():
                tissue = split_high_degree_vertices(
                    copy.deepcopy(tissue),
                    split_length=self._split_length.value(),
                )

            # 3. Solve
            mu     = None if self._mu_auto.isChecked() else self._mu_value.value()
            excl   = self._solve_excl_border.isChecked()
            margin = self._solve_border_margin.value()
            idx    = self._solver_type.currentIndex()
            def _unwrap(r):
                return r.best_result if (r and hasattr(r, "best_result")) else r

            if idx == 0:
                result = _unwrap(solve_bayesian(tissue, mu=mu,
                                                exclude_border_edges=excl,
                                                border_margin=margin))
            elif idx == 1:
                result = _unwrap(solve_bayesian_3d(tissue, mu=mu,
                                                   exclude_border_edges=excl,
                                                   border_margin=margin))
            else:
                result = solve_laplace(tissue, exclude_border_edges=excl,
                                        border_margin=margin)

            # 4. Optional geometry
            if self._do_batchelor.isChecked() and result:
                result = calculate_batchelor_stress(tissue, result)

            return labels, gray, tissue, result

        def _after(payload):
            labels, gray, tissue, result = payload
            self._labels = labels
            self._gray   = gray
            self._tissue = tissue
            self._result = result
            n = int(labels.max()) if labels is not None else 0
            ne = len(tissue.E)
            ns = int(np.sum(np.isfinite(result.tensions))) if result else 0
            self._log_msg(
                f"✓ Full pipeline — {n} cells, {ne} edges, {ns} solved.")
            self._upsert_layer("image",  self._LAYER_IMAGE,  gray,   {"colormap": "gray"})
            self._upsert_layer("labels", self._LAYER_LABELS, labels, {})
            self._refresh_topology_layer()
            self._refresh_tension_layer()

        self._start_worker(_pipeline, label="Full pipeline…", on_done=_after)

    # ── Napari layer management ───────────────────────────────────────────────

    def _upsert_layer(self, kind: str, name: str, data, kwargs: dict):
        """Add or update a napari layer, keeping existing one if present."""
        try:
            if name in self._viewer.layers:
                layer = self._viewer.layers[name]
                if kind == "shapes":
                    layer.data = data
                    for key in ("edge_color", "face_color", "edge_width",
                                "opacity"):
                        if key in kwargs:
                            setattr(layer, key, kwargs[key])
                elif kind == "vectors":
                    layer.data = data
                    for key in ("edge_color", "length", "opacity"):
                        if key in kwargs:
                            setattr(layer, key, kwargs[key])
                else:
                    layer.data = data
                    for key in ("colormap", "opacity", "scale", "translate"):
                        if key in kwargs:
                            setattr(layer, key, kwargs[key])
            else:
                kwargs["name"] = name
                if kind == "image":
                    self._viewer.add_image(data, **kwargs)
                elif kind == "labels":
                    self._viewer.add_labels(data, **kwargs)
                elif kind == "shapes":
                    self._viewer.add_shapes(data, **kwargs)
                elif kind == "vectors":
                    self._viewer.add_vectors(data, **kwargs)
        except Exception:
            self._log_msg(f"⚠ Layer update warning:\n{traceback.format_exc(limit=3)}")
