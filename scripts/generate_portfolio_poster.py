#!/opt/homebrew/bin/python3.10
"""
generate_portfolio_poster.py
────────────────────────────────────────────────────────────────────────────
Portfolio poster for ForceInferencePy.

Runs the real pipeline (segment → topology → solve → stress) and draws
every panel natively with a dark theme.

Usage:
    python scripts/generate_portfolio_poster.py [--out docs/portfolio_poster.png] [--dpi 200]
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from force_inference import segmentation, solvers, geometry, visualization
from force_inference.topology_label import extract_topology_label
from force_inference.split_four_way import split_high_degree_vertices
from force_inference.visualization import _fix_edge_pixels

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#040810"
PANEL  = "#0b1020"
DARK   = "#060d1a"
BORDER = "#1a2540"
HDR    = "#06101e"
WHITE  = "#eef2ff"
DIM    = "#4a566a"
DIM2   = "#7e8fa4"
CYAN   = "#00e5ff"
PINK   = "#ff4081"
GREEN  = "#69ff47"
GOLD   = "#ffd740"
ORANGE = "#ff9100"
PURPLE = "#ce93d8"
BLUE   = "#4d9eff"

# ── Utility ───────────────────────────────────────────────────────────────────

def _shadow(lw=2.5, fg=BG):
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def _style(ax, fc=PANEL):
    ax.set_facecolor(fc)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])


def _title(ax, main, sub="", mc=CYAN, sc=DIM2):
    ax.text(0.014, 0.984, main, transform=ax.transAxes, color=mc,
            fontsize=9.5, fontweight="bold", va="top", zorder=9,
            path_effects=_shadow(3))
    if sub:
        ax.text(0.014, 0.948, sub, transform=ax.transAxes, color=sc,
                fontsize=6.8, va="top", zorder=9, path_effects=_shadow(2))


def _label_rgb(labels, seed=7, sat=0.65, val=0.88):
    n = int(labels.max())
    if n == 0:
        return np.full((*labels.shape, 3), 0.06)
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, max(n, 1), endpoint=False)
    rng.shuffle(hues)
    hsv = np.column_stack([hues, np.full(n, sat), np.full(n, val)])
    rgb = mcolors.hsv_to_rgb(hsv)
    pal = np.zeros((n+1, 3))
    pal[0] = [0.04, 0.04, 0.07]
    pal[1:] = rgb
    return pal[labels]


def _fill_gaps(labels):
    if not np.any(labels == 0):
        return labels
    filled = labels.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    m = filled == 0
    filled[m] = filled[iy[m], ix[m]]
    return filled


def _draw_edges(ax, tissue, tensions=None, lw=1.8, alpha=0.95):
    if tensions is not None:
        finite = np.isfinite(tensions)
        vmin = np.nanpercentile(tensions[finite], 2) if finite.any() else 0
        vmax = np.nanpercentile(tensions[finite], 98) if finite.any() else 1
        if vmax <= vmin: vmax = vmin + 1e-9
        cm = plt.get_cmap("turbo")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    segs, colors = [], []
    for i, (v1, v2) in enumerate(tissue.E):
        p1 = tissue.V[v1, :2]; p2 = tissue.V[v2, :2]
        px = tissue.E_pixels[i] if tissue.E_pixels is not None else None
        if px is not None and len(px) > 1:
            try: seg = _fix_edge_pixels(px, v1_pos=p1, v2_pos=p2)
            except: seg = np.array([p1, p2])
        else:
            seg = np.array([p1, p2])
        segs.append(seg)
        if tensions is not None:
            t = tensions[i]
            c = cm(norm(t)) if np.isfinite(t) else (0.25, 0.25, 0.25, 0.3)
            colors.append(c)
        else:
            colors.append((1, 1, 1, 0.8))

    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw,
                                      capstyle="round", joinstyle="round",
                                      alpha=alpha, zorder=3))
    n_in = getattr(tissue, "num_inner_vertices", len(tissue.V))
    ax.scatter(tissue.V[:n_in, 0], tissue.V[:n_in, 1],
               s=np.pi * lw**2 * 2, c="white", edgecolors="none",
               zorder=4, alpha=0.65)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(img_path):
    print("  Segmenting…", flush=True)
    labels, gray = segmentation.segment_grayscale(
        str(img_path), h_depth=2.0, min_cell_size=5)

    print("  Topology…", flush=True)
    tissue = extract_topology_label(
        labels, min_edge_len=1,
        use_skeleton_geometry=False,
        collapse_stubs=False,
        collapse_tiny_twins=False,
    )
    tissue = split_high_degree_vertices(copy.deepcopy(tissue), split_length=4.0)

    print("  Solving…", flush=True)
    bayes = solvers.solve_bayesian(tissue, mu=1e-2)

    print("  Stress…", flush=True)
    result_s = geometry.calculate_batchelor_stress(
        copy.deepcopy(tissue), copy.deepcopy(bayes))

    return labels, gray, tissue, bayes, result_s


# ── Individual panel content ──────────────────────────────────────────────────

def _show_membrane(ax, gray):
    _style(ax)
    img = gray.astype(float)
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    # gamma lift to make membranes pop
    img = img ** 0.6
    ax.imshow(img, cmap="gray", aspect="auto", origin="upper")
    _title(ax, "Fluorescence input", "raw membrane image  ·  contrast enhanced", mc=WHITE)


def _show_segmentation(ax, labels):
    _style(ax)
    ax.imshow(_label_rgb(_fill_gaps(labels)), aspect="auto", origin="upper")
    n = int(labels.max())
    _title(ax, "Cell segmentation", f"{n} cells  ·  watershed + h-minima", mc=GREEN)


def _show_topology(ax, labels, tissue):
    _style(ax)
    H, W = labels.shape
    # Bright enough to orient the eye, dark enough that edges pop
    ax.imshow(_label_rgb(_fill_gaps(labels)) * 0.45, aspect="auto", origin="upper")
    ax.autoscale(False)
    _draw_edges(ax, tissue, tensions=None, lw=1.8)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    nv, ne = len(tissue.V), len(tissue.E)
    _title(ax, "Junction topology",
           f"{nv} vertices  ·  {ne} edges  ·  3-way only", mc=GOLD)


def _show_tensions_hero(ax, labels, tissue, bayes):
    """Large hero panel — tensions on dimmed cells, full image."""
    _style(ax, fc=DARK)
    H, W = labels.shape
    ax.imshow(_label_rgb(_fill_gaps(labels)) * 0.28, aspect="auto", origin="upper")
    ax.autoscale(False)
    _draw_edges(ax, tissue, tensions=bayes.tensions, lw=2.0)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)

    # Inline colorbar
    cax = ax.inset_axes([0.67, 0.022, 0.31, 0.022])
    finite = np.isfinite(bayes.tensions)
    vmin = np.nanpercentile(bayes.tensions[finite], 2)
    vmax = np.nanpercentile(bayes.tensions[finite], 98)
    sm = plt.cm.ScalarMappable(cmap="turbo",
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cb.set_ticklabels(["low", "", "high"])
    cb.ax.tick_params(colors=DIM2, labelsize=5.5, length=0)
    cb.outline.set_edgecolor(BORDER)

    nsolved = int(np.sum(finite))
    _title(ax, "Bayesian membrane tensions",
           f"{nsolved} edges solved  ·  evidence-maximised regularisation μ",
           mc=PINK, sc=DIM2)


def _show_stress(ax, labels, tissue, result_s):
    _style(ax)
    H, W = labels.shape
    # Show dim cells + tension-coloured edges + stress crosses for maximum info density
    ax.imshow(_label_rgb(_fill_gaps(labels)) * 0.35, aspect="auto", origin="upper")
    ax.autoscale(False)
    # Thin tension edges for context
    _draw_edges(ax, tissue, tensions=result_s.tensions, lw=1.2, alpha=0.55)
    # Principal stress crosses on top
    visualization.plot_cell_stress_crosses(ax, tissue, result_s, scale=1.2)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    _title(ax, "Batchelor cell stress",
           "principal axes  ·  tension (warm) / compression (cool)", mc=ORANGE)


# ── Feature cards ─────────────────────────────────────────────────────────────

FEATURES = [
    ("⚙", GOLD,   "Bayesian\nSolver",
     "Evidence maximisation\nover log-reg μ\nsparse LSQR system",
     "solve_bayesian"),
    ("κ",   CYAN,   "Young-\nLaplace",
     "ΔP = Tκ  curved edge\narc circle fits +\nanalytic tangents",
     "solve_laplace"),
    ("⬡",  GREEN,  "3D Force\nBalance",
     "Newell cell normals\ncross-product pressure\nauto-fallback 2D",
     "solve_bayesian_3d"),
    ("↻",  PINK,   "Time-Series\nAlign",
     "Shared-edge scale\nlog-ratio minimisation\nfluorescence calib.",
     "TimeSeries.align"),
    ("σ",  ORANGE, "Batchelor\nStress",
     "Per-cell 2×2 tensor\nvectorised einsum\nscatter-add",
     "calculate_batchelor\n_stress"),
    ("⬆",  PURPLE, "2.5D / 3D\nExtension",
     "Z from brightest\nconfocal voxel  ·\nfull 3D solver",
     "map_z_to_vertices"),
]


def _draw_features(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    n = len(FEATURES)
    W = 1.0 / n

    for i, (icon, col, title, desc, api) in enumerate(FEATURES):
        xl, xr = i*W + 0.006, (i+1)*W - 0.006
        xc = (xl + xr) / 2
        h  = 0.92

        # Card
        ax.add_patch(FancyBboxPatch((xl, 0.04), xr-xl, h,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.012",
                     fc=PANEL, ec=BORDER, lw=0.8, zorder=1))

        # Accent stripe
        ax.add_patch(FancyBboxPatch((xl, 0.04+h-0.11), xr-xl, 0.11,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.008",
                     fc=col, ec="none", alpha=0.18, zorder=2))

        # Icon
        ax.text(xc, 0.04+h*0.79, icon, transform=ax.transAxes,
                color=col, fontsize=21, va="center", ha="center",
                zorder=3, path_effects=_shadow(2, PANEL))

        # Title
        ax.text(xc, 0.04+h*0.60, title, transform=ax.transAxes,
                color=WHITE, fontsize=8, fontweight="bold",
                va="center", ha="center", linespacing=1.2, zorder=3)

        # Description
        ax.text(xc, 0.04+h*0.34, desc, transform=ax.transAxes,
                color=DIM2, fontsize=6.5, va="center", ha="center",
                linespacing=1.4, zorder=3)

        # API chip
        ax.add_patch(FancyBboxPatch((xl+0.005, 0.055), xr-xl-0.010, 0.11,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.005",
                     fc=col, ec="none", alpha=0.15, zorder=3))
        ax.text(xc, 0.110, api, transform=ax.transAxes,
                color=col, fontsize=5.8, va="center", ha="center",
                fontfamily="monospace", zorder=4)


# ── Code block ────────────────────────────────────────────────────────────────

def _draw_code(ax):
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
        sp.set_linewidth(0.9)
    ax.set_xticks([]); ax.set_yticks([])

    # Title bar
    ax.add_patch(FancyBboxPatch((0, 0.89), 1, 0.11,
                 transform=ax.transAxes, boxstyle="square,pad=0",
                 fc="#161b22", ec="none", zorder=2))
    for x, c in [(0.025, "#ff5f56"), (0.075, "#ffbd2e"), (0.125, "#27c93f")]:
        ax.text(x, 0.945, "●", transform=ax.transAxes, color=c,
                fontsize=9, va="center", zorder=3)
    ax.text(0.5, 0.945, "quick_start.py", transform=ax.transAxes,
            color=DIM, fontsize=7, va="center", ha="center", zorder=3)

    lines = [
        [("import ", BLUE), ("force_inference", CYAN), (" as fi", WHITE)],
        [],
        [("# 1 · segment", DIM)],
        [("labels", WHITE), (", gray = fi.", WHITE),
         ("segment_grayscale", CYAN), ("(img_path)", WHITE)],
        [],
        [("# 2 · topology + 4-way split", DIM)],
        [("tissue", WHITE), (" = fi.", WHITE),
         ("extract_topology_label", CYAN), ("(labels)", WHITE)],
        [("tissue", WHITE), (" = ", WHITE),
         ("split_high_degree_vertices", CYAN), ("(tissue)", WHITE)],
        [],
        [("# 3 · solve Bayesian", DIM)],
        [("result", WHITE), (" = fi.", WHITE),
         ("solve_bayesian", CYAN), ("(tissue)", WHITE),
         (".best_result", DIM)],
        [],
        [("# 4 · visualise", DIM)],
        [("print", BLUE), ("(result.", WHITE),
         ("summary", CYAN), ("())", WHITE)],
        [("fi.", WHITE), ("plot_tensions", CYAN),
         ("(ax, tissue, result)", WHITE)],
    ]

    # char_advance: monospace DejaVu at fontsize 7.0 in a ~8.64" wide axes
    # 1 char ≈ 7.0/72 * 0.6 (aspect) / 8.64 ≈ 0.00337 axes-fraction per char
    # empirically, 0.014 gives good spacing at 200 DPI
    CHAR_ADV = 0.0138
    y0, dh = 0.855, 0.058
    for li, spans in enumerate(lines):
        y = y0 - li * dh
        if y < 0.02 or not spans:
            continue
        x = 0.025
        for text, color in spans:
            ax.text(x, y, text, transform=ax.transAxes,
                    color=color, fontsize=7.0, va="top",
                    fontfamily="monospace", zorder=4)
            x += len(text) * CHAR_ADV


# ── Pipeline strip ────────────────────────────────────────────────────────────

def _draw_pipeline(ax):
    ax.set_facecolor(HDR)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    steps = [
        ("01", "Segment",        "segment_grayscale\n/ segment_cellpose",     GOLD),
        ("02", "Topology",       "extract_topology_label\n+ split_four_way",  CYAN),
        ("03", "Curvature",      "compute_curvature\narc fit + tangents",     GREEN),
        ("04", "Bayesian Solve", "solve_bayesian\nevidence max. μ scan",      PINK),
        ("05", "Stress",         "calculate_batchelor\n_stress",              ORANGE),
        ("06", "Time-series",    "TimeSeries.align\nshared-edge log-ratio",   PURPLE),
    ]

    n = len(steps)
    xs = np.linspace(0.055, 0.945, n)

    for i, (num, name, detail, col) in enumerate(steps):
        xc = xs[i]; bw = 0.12

        # Arrow
        if i > 0:
            ax.annotate("", xy=(xc - bw/2, 0.50),
                        xytext=(xs[i-1] + bw/2, 0.50),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.5",
                                        color=BORDER, lw=1.1))

        ax.add_patch(FancyBboxPatch((xc-bw/2, 0.06), bw, 0.88,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.012",
                     fc=PANEL, ec=col, lw=1.4, zorder=2))

        # Badge
        ax.add_patch(FancyBboxPatch((xc-bw/2+0.005, 0.76), 0.022, 0.16,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.003",
                     fc=col, ec="none", zorder=3))
        ax.text(xc-bw/2+0.016, 0.84, num,
                transform=ax.transAxes, color=BG, fontsize=5.5,
                fontweight="bold", va="center", ha="center", zorder=4)

        ax.text(xc, 0.58, name,
                transform=ax.transAxes, color=WHITE, fontsize=7.5,
                fontweight="bold", va="center", ha="center", zorder=3)

        ax.text(xc, 0.24, detail,
                transform=ax.transAxes, color=DIM2, fontsize=5.8,
                va="center", ha="center", linespacing=1.35,
                fontfamily="monospace", zorder=3)


# ── Header ────────────────────────────────────────────────────────────────────

def _draw_header(ax, n_cells, n_edges, n_solved):
    ax.set_facecolor(HDR)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Gradient
    g = np.linspace(0, 1, 512).reshape(1, -1)
    ax.imshow(g, aspect="auto", extent=[0,1,0,1], transform=ax.transAxes,
              cmap=LinearSegmentedColormap.from_list("h",[HDR,"#081328"]),
              zorder=0, alpha=0.6)

    # Cyan left bar
    ax.add_patch(mpatches.Rectangle((0,0), 0.0022, 1,
                 transform=ax.transAxes, fc=CYAN, ec="none", zorder=5,
                 clip_on=True))

    ax.text(0.012, 0.82, "ForceInferencePy",
            transform=ax.transAxes, color=WHITE,
            fontsize=27, fontweight="bold", va="top",
            clip_on=True,
            path_effects=_shadow(4, HDR))

    ax.text(0.012, 0.18,
            "Bayesian cell force inference from fluorescence microscopy  ·  "
            "label-driven topology  ·  3D Newell force balance  ·  time-series scale alignment",
            transform=ax.transAxes, color=DIM2, fontsize=9.5, va="bottom",
            clip_on=True)

    # Stats (4 chips: cells / edges / solved / speed)
    stat_items = [
        (f"{n_cells}",  "cells",    CYAN),
        (f"{n_edges}",  "edges",    GOLD),
        (f"{n_solved}", "solved",   GREEN),
        ("<5 ms",       "per edge", PINK),
    ]
    stat_x0 = 0.585
    stat_w  = 0.048
    stat_gap = 0.008
    for i, (val, lbl, col) in enumerate(stat_items):
        xc = stat_x0 + i * (stat_w + stat_gap)
        ax.add_patch(FancyBboxPatch((xc, 0.10), stat_w, 0.80,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.010",
                     fc=col, ec="none", alpha=0.14, zorder=3, clip_on=True))
        ax.text(xc + stat_w/2, 0.66, val,
                transform=ax.transAxes, color=col, fontsize=13,
                fontweight="bold", va="center", ha="center",
                zorder=4, clip_on=True, path_effects=_shadow(2, HDR))
        ax.text(xc + stat_w/2, 0.24, lbl,
                transform=ax.transAxes, color=DIM2, fontsize=6.5,
                va="center", ha="center", zorder=4, clip_on=True)

    # Compact badges – 3 chips that fit inside [0,1] x-range
    badges = [("Py 3.8–3.12", BLUE), ("MIT", GREEN), ("CI ✓", PINK)]
    badge_w = 0.040
    badge_gap = 0.006
    bx = 0.843
    for txt, col in badges:
        if bx + badge_w > 0.984:
            break
        ax.add_patch(FancyBboxPatch((bx, 0.53), badge_w, 0.38,
                     transform=ax.transAxes,
                     boxstyle="round,pad=0.006",
                     fc=col, ec="none", alpha=0.20, zorder=3, clip_on=True))
        ax.text(bx + badge_w/2, 0.720, txt,
                transform=ax.transAxes, color=col, fontsize=6.2,
                va="center", ha="center", fontweight="bold",
                zorder=4, clip_on=True)
        bx += badge_w + badge_gap

    ax.text(0.985, 0.12,
            "github.com/Weiykong/ForceInferencePy",
            transform=ax.transAxes, color=DIM,
            fontsize=7.5, va="bottom", ha="right", clip_on=True)


# ── Master ────────────────────────────────────────────────────────────────────

def make_poster(img_path: Path, out_path: Path, dpi: int = 200) -> None:
    print("Running pipeline…")
    labels, gray, tissue, bayes, result_s = _run_pipeline(img_path)
    nc = int(labels.max())
    ne = len(tissue.E)
    ns = int(np.sum(np.isfinite(bayes.tensions)))
    print(f"  {nc} cells · {ne} edges · {ns} solved")

    # ── Figure layout ─────────────────────────────────────────────────────
    #
    # 24" × 14.2"
    #
    #  [  Header (1.3")                                                     ]
    #  [ Hero tension | small-2x2 (segm, membr, topol, stress) ] (8.0")
    #  [ Feature cards (6)                                      ] (2.9")
    #  [ Pipeline strip (left 62%) | Code snippet (right 38%)  ] (1.8")
    #  [ Footer (0.2")                                          ]
    #
    fig = plt.figure(figsize=(24.0, 14.2), facecolor=BG)

    gs = gridspec.GridSpec(
        5, 1,
        figure=fig,
        height_ratios=[1.3, 8.0, 2.9, 1.8, 0.2],
        hspace=0.022,
        left=0.010, right=0.990,
        top=0.990, bottom=0.010,
    )

    # ── Header ────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[0])
    _draw_header(ax_hdr, nc, ne, ns)

    # ── Image row: hero left + 2×2 right ──────────────────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], wspace=0.018,
        width_ratios=[52, 48],
    )
    ax_hero = fig.add_subplot(gs1[0])
    _show_tensions_hero(ax_hero, labels, tissue, bayes)

    gs2 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs1[1], hspace=0.018, wspace=0.018)
    ax_seg   = fig.add_subplot(gs2[0, 0]); _show_segmentation(ax_seg, labels)
    ax_mem   = fig.add_subplot(gs2[0, 1]); _show_membrane(ax_mem, gray)
    ax_topo  = fig.add_subplot(gs2[1, 0]); _show_topology(ax_topo, labels, tissue)
    ax_str   = fig.add_subplot(gs2[1, 1]); _show_stress(ax_str, labels, tissue, result_s)

    # ── Feature cards ─────────────────────────────────────────────────────
    ax_feat = fig.add_subplot(gs[2])
    _draw_features(ax_feat)

    # ── Pipeline + Code ───────────────────────────────────────────────────
    gs3 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[3], wspace=0.018,
        width_ratios=[62, 38])
    _draw_pipeline(fig.add_subplot(gs3[0]))
    _draw_code(fig.add_subplot(gs3[1]))

    # ── Footer ────────────────────────────────────────────────────────────
    ax_foot = fig.add_subplot(gs[4])
    ax_foot.set_facecolor(PANEL)
    for sp in ax_foot.spines.values(): sp.set_visible(False)
    ax_foot.set_xticks([]); ax_foot.set_yticks([])
    ax_foot.text(0.012, 0.5,
                 "ForceInferencePy  ·  Python · NumPy · SciPy · scikit-image · matplotlib",
                 transform=ax_foot.transAxes, color=DIM, fontsize=7, va="center")
    ax_foot.text(0.988, 0.5, "weiyuankong@gmail.com",
                 transform=ax_foot.transAxes, color=DIM, fontsize=7,
                 va="center", ha="right")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=REPO/"data"/"test.tif")
    ap.add_argument("--out",   type=Path, default=REPO/"docs"/"portfolio_poster.png")
    ap.add_argument("--dpi",   type=int,  default=200)
    args = ap.parse_args()
    make_poster(args.image, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
