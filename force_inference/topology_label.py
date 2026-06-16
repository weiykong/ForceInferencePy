"""
Label-Driven Topology Extraction for Force Inference.

PURE LABEL APPROACH — no skeleton dependency for topology.

Key insight: the skeleton merges close junctions (twin junctions) because
skeletonization is a geometric operation that cannot preserve topological
features smaller than ~2 px. This module detects vertices and edges
directly from the 8-neighbourhood label pattern, guaranteeing that every
real cell-cell interface becomes an edge.

Algorithm:
  1. Classify every boundary pixel by its 3×3 neighbourhood:
       - VERTEX pixel: ≥3 distinct nonzero labels meet (triple+ point)
       - EDGE pixel:   exactly 2 nonzero labels meet (interface pixel)
  2. Connected-component clustering of VERTEX pixels → vertices
  3. For each cell pair (c1, c2):
       a. Collect all EDGE pixels tagged (c1, c2)
       b. Split into connected components
       c. For each component, find which vertex clusters it touches
       d. Each component with two distinct vertex endpoints → one edge
  4. (Optional) Snap vertex positions onto skeleton for sub-pixel geometry
  5. Order edge pixels from v1 → v2, build Tissue

This approach is used by TissueAnalyzer, MorphographX, SEGGA and other
robust tissue-topology libraries.
"""

import numpy as np
from skimage import morphology, measure
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import logging
from typing import Optional, List, Tuple, Dict, Set

from .core import Tissue

logger = logging.getLogger("ForceInference.TopologyLabel")


# =============================================================================
# Public API
# =============================================================================

def extract_topology_label(
    labels: np.ndarray,
    *,
    min_edge_len: int = 2,
    trace_pixels: bool = True,
    clean: bool = False,
    min_clean_edge_len: float = 3.0,
    remove_outer_layer: bool = False,
    vertex_cluster_r: float = 2.0,
    use_skeleton_geometry: bool = True,
    curve_points: int = 0,
    collapse_stubs: bool = True,
    stub_edge_threshold: float = 3.0,
    collapse_tiny_twins: bool = False,
    tiny_twin_threshold: float = 3.0,
    junction_window: int = 0,
    dilate_labels: int = 0,
) -> Optional[Tissue]:
    """
    Extract tissue topology purely from label-neighbourhood patterns.

    Args:
        labels:              Segmented label image (int). Background = 0.
        min_edge_len:        Minimum number of pixels in an edge to keep it.
        trace_pixels:        Store ordered pixel paths in tissue.E_pixels.
        clean:               Collapse very short edges after extraction.
        min_clean_edge_len:  Threshold for edge collapse (only if clean=True).
        remove_outer_layer:  Zero out cells touching the image border.
        vertex_cluster_r:    Radius for merging nearby vertex pixels within
                             one connected component.  Does NOT merge across
                             components, so twin junctions stay separate.
        use_skeleton_geometry: If True, snap vertex positions onto the
                             skeletonized membrane for sub-pixel accuracy.
        curve_points:        If > 0, resample each edge to this many points
                             using spline interpolation.
        collapse_stubs:      If True (default), collapse short stub edges
                             where one endpoint has degree ≤ 2. This removes
                             spurious short edges from wide junction blobs
                             without destroying real twin-junction short edges
                             (which have degree ≥ 3 at both ends).
        stub_edge_threshold: Maximum edge length (px) to consider for stub
                             collapse. Only edges shorter than this with a
                             low-degree endpoint are collapsed.
        collapse_tiny_twins: If True (default), promote below-resolution
                             twin junctions to 4-way vertices.  After stub
                             collapse, any remaining short edge where BOTH
                             endpoints have degree ≥ 3 is below the image
                             resolution limit (the two 3-way junctions can't
                             be distinguished at this scale).  Merging them
                             into one 4-way vertex at the midpoint is the
                             correct representation at that resolution — not
                             an approximation.
        tiny_twin_threshold: Edge length (px) below which a high-degree
                             twin-junction edge is treated as below-resolution
                             and collapsed to a 4-way vertex.  Default 3 px.
                             Only edges where BOTH endpoints have degree ≥ 3
                             are candidates; single-pixel edges between small
                             cells are typically real and should not be merged.
                             Increase to 4–5 only for very blurry images where
                             true sub-pixel twins are known to exist.
        junction_window:     Half-size of the analysis window for vertex
                             detection.  0 = auto (adapts to membrane
                             thickness).  1 = 3×3 window (default for thin
                             membranes).  2 = 5×5 (for thick/blurry
                             membranes like test.tif).
        dilate_labels:       Number of pixels to expand cell labels into
                             background BEFORE junction detection.  This
                             closes the membrane gap at junctions so that
                             the analysis window can see enough cells.
                             0 = auto (measures membrane width and adapts).
                             Set to 1-2 for thick membranes.

    Returns:
        Tissue object, or None if extraction fails.
    """
    logger.info("Extracting topology (Label-Driven)...")

    labels_proc = labels.copy().astype(np.int32)
    H, W = labels_proc.shape

    # ------------------------------------------------------------------ #
    # 0.  Pre-processing                                                  #
    # ------------------------------------------------------------------ #
    if remove_outer_layer:
        bm = np.zeros_like(labels_proc, dtype=bool)
        bm[0, :] = bm[-1, :] = bm[:, 0] = bm[:, -1] = True
        border_labels = np.unique(labels_proc[bm])
        labels_proc[np.isin(labels_proc, border_labels)] = 0

    unique_labels = np.unique(labels_proc)
    if len(unique_labels) < 3:  # need at least background + 2 cells
        logger.warning("Fewer than 2 cell labels found.")
        return None

    # ------------------------------------------------------------------ #
    # 0b. Auto-detect membrane thickness and adapt parameters              #
    # ------------------------------------------------------------------ #
    membrane_width = _estimate_membrane_width(labels_proc)
    logger.info(f"Estimated membrane width: {membrane_width:.1f} px")

    # Auto-select dilation amount
    effective_dilate = dilate_labels
    if effective_dilate == 0:
        # Auto: dilate by half the membrane width (rounded up),
        # capped so we don't merge cells.
        effective_dilate = min(int(np.ceil(membrane_width / 2)), 3)
    if effective_dilate < 1:
        effective_dilate = 1

    # Auto-select junction window
    effective_window = junction_window
    if effective_window == 0:
        # Auto: use 5×5 for thick membranes (width > 2), else 3×3
        effective_window = 2 if membrane_width > 2.0 else 1

    # ------------------------------------------------------------------ #
    # 0c.  Full Voronoi — fill every background pixel with nearest cell   #
    # ------------------------------------------------------------------ #
    # With no background (zero) pixels, the 3×3 window at every boundary
    # pixel sees the correct set of cell labels regardless of membrane
    # thickness.  Large diffuse junction blobs no longer produce gaps.
    labels_expanded = _full_voronoi_labels(labels_proc)
    logger.info("Full Voronoi label expansion applied")

    # ------------------------------------------------------------------ #
    # 1-3.  Detect vertices and build edges                               #
    # ------------------------------------------------------------------ #
    # Primary path: corner-based classification.  This is more robust for
    # thick or blurry membranes because each inter-pixel corner always sees
    # a local 2x2 cell configuration after Voronoi fill.
    vertex_corner_mask, edge_corner_map = _classify_boundary_corners(
        labels_expanded
    )
    n_vcorners = int(np.sum(vertex_corner_mask))
    n_ecorners = int(np.sum(edge_corner_map > 0))
    logger.info(
        f"Boundary corners: {n_vcorners} vertex, {n_ecorners} edge"
    )

    edges_list = []
    edges_cells = []
    edges_pixels = []
    vertices = np.zeros((0, 2), dtype=float)
    vertex_cell_sets: List[Set[int]] = []

    if n_vcorners > 0:
        vertices, vertex_cell_sets, corner_to_vid = _cluster_vertex_corners(
            vertex_corner_mask, labels_expanded
        )
        logger.info(f"Corner vertices: {len(vertices)}")

        if len(vertices) >= 2:
            if use_skeleton_geometry:
                boundary = _labels_to_boundary(labels_proc)
                skel = morphology.skeletonize(boundary)
                vertices = _snap_to_skeleton(vertices, skel)

            # Compute adaptive search radii from typical cell size.
            # Average cell diameter ≈ 2 * sqrt(cell_area / π).
            _n_cells = max(1, len(np.unique(labels_proc)) - 1)
            _avg_cell_r = np.sqrt(H * W / (_n_cells * np.pi))
            # Radii: large enough to catch long edges but capped at 1.5× the
            # average cell radius to avoid connecting wrong vertex pairs.
            _fallback_r = float(np.clip(_avg_cell_r * 1.5, 20.0, 80.0))
            _recovery_r = float(np.clip(_avg_cell_r * 1.5, 20.0, 80.0))

            result = _build_edges_from_corners(
                edge_corner_map,
                vertex_corner_mask,
                corner_to_vid,
                vertices,
                vertex_cell_sets,
                labels_expanded,
                min_edge_len,
                len(vertices),
                fallback_radius=_fallback_r,
                recovery_radius=_recovery_r,
            )
            edges_list = result['edges']
            edges_cells = result['cells']
            edges_pixels = result['pixels']
            vertices = result['vertices']
            vertex_cell_sets = result['vertex_cell_sets']
            logger.info(f"Corner edges: {len(edges_list)}")

    # Fallback: legacy pixel-window detector.
    if len(edges_list) == 0:
        logger.info("Corner detector insufficient, falling back to pixel detector")

        vertex_mask, edge_mask, edge_pair_map = _classify_boundary_pixels(
            labels_expanded, half_window=effective_window
        )
        n_vpx = int(np.sum(vertex_mask))
        n_epx = int(np.sum(edge_mask))
        logger.info(f"Boundary pixels: {n_vpx} vertex, {n_epx} edge")

        if n_vpx == 0:
            logger.warning("No vertex pixels found.")
            return None

        vertices, vertex_cell_sets = _cluster_vertex_pixels(
            vertex_mask, labels_expanded, vertex_cluster_r
        )
        logger.info(f"Pixel vertices: {len(vertices)}")

        if len(vertices) < 2:
            logger.warning("Fewer than 2 vertices found.")
            return None

        if use_skeleton_geometry:
            boundary = _labels_to_boundary(labels_proc)
            skel = morphology.skeletonize(boundary)
            vertices = _snap_to_skeleton(vertices, skel)

        vertex_pixel_map = _build_vertex_pixel_map(
            vertex_mask, labels_expanded, vertices, vertex_cell_sets
        )

        result = _build_edges_from_labels(
            edge_mask, edge_pair_map, vertex_mask, vertex_pixel_map,
            vertices, vertex_cell_sets, labels_expanded, min_edge_len
        )
        edges_list = result['edges']
        edges_cells = result['cells']
        edges_pixels = result['pixels']
        vertices = result['vertices']
        vertex_cell_sets = result['vertex_cell_sets']
        logger.info(f"Pixel edges: {len(edges_list)}")

    if len(edges_list) == 0:
        logger.warning("No edges found.")
        return None

    # ------------------------------------------------------------------ #
    # 4.  Build arrays and sort vertices (inner first, border last)       #
    # ------------------------------------------------------------------ #
    E = np.array(edges_list, dtype=int)
    E_cells = np.array(edges_cells, dtype=int)

    # Sort vertices: inner first, border last (solver requirement)
    margin = 2.0
    is_border_v = (
        (vertices[:, 0] <= margin) | (vertices[:, 0] >= W - margin) |
        (vertices[:, 1] <= margin) | (vertices[:, 1] >= H - margin)
    )
    inner_idx = np.where(~is_border_v)[0]
    border_idx = np.where(is_border_v)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner = len(inner_idx)

    vertices = vertices[sorted_idx]
    old_to_new = np.empty(len(sorted_idx), dtype=int)
    old_to_new[sorted_idx] = np.arange(len(sorted_idx))
    E = old_to_new[E]

    # Also remap edge pixels vertex references
    E_pixels_ordered = []
    if trace_pixels:
        for eidx, (v1, v2) in enumerate(E):
            pts = edges_pixels[eidx]
            ordered = _order_pixels(pts, vertices[v1], vertices[v2])
            if curve_points > 0 and len(ordered) >= 4:
                ordered = _resample_spline(ordered, vertices[v1],
                                           vertices[v2], curve_points)
            E_pixels_ordered.append(ordered)

    V = np.column_stack((vertices, np.zeros(len(vertices))))

    # ------------------------------------------------------------------ #
    # 5.  Centroids + C_v                                                 #
    # ------------------------------------------------------------------ #
    props = measure.regionprops(labels_proc)
    max_lbl = int(labels_proc.max())
    C_centroids = np.zeros((max_lbl, 2), dtype=float)
    for p in props:
        if p.label > 0:
            C_centroids[p.label - 1] = p.centroid[::-1]

    C_v = _build_C_v(labels_proc, V, E, E_cells)

    tissue = Tissue(V, E, E_cells, C_centroids, C_v, labels_proc)
    tissue.num_inner_vertices = num_inner
    if trace_pixels:
        tissue.E_pixels = E_pixels_ordered

    # ------------------------------------------------------------------ #
    # 6.  Collapse stub edges (degree-aware)                              #
    # ------------------------------------------------------------------ #
    #     A "stub" is a short edge where one endpoint has degree ≤ 2.
    #     This means it is NOT a real twin junction (which has degree ≥ 3
    #     at both ends).  It is an artifact of a wide junction blob being
    #     split into multiple sub-vertices.
    #
    #     We collapse stubs by merging the low-degree endpoint into its
    #     high-degree neighbor, preserving the real topology.
    if collapse_stubs and len(tissue.E) > 0:
        n_before = len(tissue.E)
        tissue = _collapse_stub_edges(tissue, stub_edge_threshold)
        n_after = len(tissue.E)
        if n_before != n_after:
            logger.info(
                f"Stub collapse: {n_before} → {n_after} edges "
                f"(removed {n_before - n_after} stubs)"
            )

    # ------------------------------------------------------------------ #
    # 6.5  Collapse below-resolution twin junctions → 4-way vertices     #
    # ------------------------------------------------------------------ #
    #     After stub removal, any remaining short edge where BOTH         #
    #     endpoints have degree ≥ 3 is a "real" twin junction that is    #
    #     too small to resolve at the image scale.  Collapsing it into   #
    #     one 4-way vertex at the midpoint is the correct representation  #
    #     at that resolution — NOT fabricating geometry as "force longer" #
    #     would do.                                                       #
    if collapse_tiny_twins and len(tissue.E) > 0:
        n_before = len(tissue.E)
        tissue = _collapse_tiny_twin_junctions(tissue, tiny_twin_threshold)
        n_after = len(tissue.E)
        if n_before != n_after:
            logger.info(
                f"Tiny-twin collapse: {n_before} → {n_after} edges "
                f"(promoted {n_before - n_after} below-resolution twins "
                f"to 4-way vertices)"
            )

    # ------------------------------------------------------------------ #
    # 7.  Optional: collapse short edges (general)                        #
    # ------------------------------------------------------------------ #
    if clean and min_clean_edge_len > 0 and len(tissue.E) > 0:
        tissue = _clean_tissue_graph(tissue, min_clean_edge_len)

    logger.info(
        f"Done: {len(tissue.V)} vertices, {len(tissue.E)} edges, "
        f"{tissue.num_inner_vertices} inner vertices."
    )
    return tissue


# =============================================================================
# Phase 0b:  Membrane thickness estimation and label expansion
# =============================================================================

def _estimate_membrane_width(labels: np.ndarray) -> float:
    """
    Estimate the typical membrane (background) width by measuring
    horizontal and vertical runs of background pixels between cells.
    """
    H, W = labels.shape
    bg = (labels == 0)

    # Sample horizontal runs
    runs = []
    for y in range(0, H, max(1, H // 50)):  # sample ~50 rows
        run = 0
        in_membrane = False
        for x in range(W):
            if bg[y, x]:
                run += 1
                in_membrane = True
            else:
                if in_membrane and 1 <= run <= 20:
                    runs.append(run)
                run = 0
                in_membrane = False

    # Sample vertical runs
    for x in range(0, W, max(1, W // 50)):
        run = 0
        in_membrane = False
        for y in range(H):
            if bg[y, x]:
                run += 1
                in_membrane = True
            else:
                if in_membrane and 1 <= run <= 20:
                    runs.append(run)
                run = 0
                in_membrane = False

    if not runs:
        return 1.0
    return float(np.median(runs))


def _voronoi_expand_labels(
    labels: np.ndarray,
    iterations: int = 1,
) -> np.ndarray:
    """
    Expand each nonzero cell label into neighbouring background pixels.

    This is a voronoi-like expansion: each background pixel gets assigned
    to the nearest cell.  We do this iteratively (one pixel per iteration)
    so that competing cells meet at their midpoint.

    The expanded labels are used ONLY for junction detection — the
    original labels remain the source of truth for edge cell assignment.

    Vectorized implementation using shifted arrays (fast for large images).
    """
    expanded = labels.copy()
    H, W = labels.shape

    # 4-connected offsets
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(iterations):
        bg_mask = (expanded == 0)
        if not np.any(bg_mask):
            break

        # For each direction, get the label from that neighbour
        pad = np.pad(expanded, 1, mode='constant', constant_values=0)
        new_labels = expanded.copy()

        for dy, dx in shifts:
            neighbour = pad[1+dy:H+1+dy, 1+dx:W+1+dx]
            # Fill background pixels with this neighbour's label
            # (only if the neighbour has a nonzero label)
            fill_mask = bg_mask & (neighbour > 0) & (new_labels == 0)
            new_labels[fill_mask] = neighbour[fill_mask]

        expanded = new_labels

    return expanded


def _full_voronoi_labels(labels: np.ndarray) -> np.ndarray:
    """
    Assign every background pixel to its nearest cell (true Voronoi).

    Unlike _voronoi_expand_labels (limited iterations), this fills the
    ENTIRE background in one call using scipy's distance_transform_edt.
    The result: at every pixel in the image, the label equals the ID of
    the nearest cell.

    This is used ONLY for junction detection.  The original labels remain
    the source of truth for geometry and cell assignment.

    Why this is needed:
      Large diffuse junction blobs (where 3-4 cells meet in a blurry ~10px
      area) have a center that is 5-10px from any cell boundary.  The
      iterative expansion capped at 2-3px never reaches that center, so
      the analysis window there only sees 0-2 cells → no VERTEX pixel →
      the whole junction is missed.  The full Voronoi assigns the correct
      cell label to every pixel in the blob, guaranteeing that the analysis
      window at the junction center sees all the surrounding cells.
    """
    from scipy.ndimage import distance_transform_edt

    bg = (labels == 0)
    if not np.any(bg):
        return labels.copy()

    # distance_transform_edt with return_indices gives, for each background
    # pixel, the coordinates of the nearest foreground (cell) pixel.
    _, nearest = distance_transform_edt(bg, return_indices=True)
    result = labels.copy()
    # Assign each background pixel the label of its nearest cell pixel
    result[bg] = labels[nearest[0][bg], nearest[1][bg]]
    return result


# =============================================================================
# Corner-based topology detection  (TissueAnalyzer / MorphographX style)
# =============================================================================
#
# Fundamental insight: instead of classifying PIXELS (which needs a tunable
# analysis window and breaks for thick/blurry membranes), classify CORNERS
# between pixels.  Each corner sees exactly 4 pixels — a 2×2 square — and
# the number of distinct cell labels in that square is exact:
#
#   1 label  → interior corner  (all same cell)
#   2 labels → EDGE corner      (boundary between two cells)
#   3+ labels → VERTEX corner   (junction of 3 or 4 cells)
#
# This requires NO tuning parameters and works correctly for any membrane
# thickness or junction size, because with a full Voronoi fill every
# background pixel is assigned to its nearest cell and every corner
# sees the correct local cell configuration.


def _classify_boundary_corners(
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify every inter-pixel corner into VERTEX or EDGE.

    Args:
        labels: Fully-filled label image (no zeros — use _full_voronoi_labels
                first).  Shape (H, W).

    Returns:
        vertex_corner_mask: bool (H-1, W-1) — True at T/Y/X-junction corners.
        edge_corner_map:    int64 (H-1, W-1) — c1*K + c2 at edge corners
                            (c1 < c2 are the two cell labels), 0 elsewhere.
    """
    H, W = labels.shape
    K = int(labels.max()) + 1

    # Four pixels around each corner — vectorized
    tl = labels[:-1, :-1]   # top-left
    tr = labels[:-1, 1:]    # top-right
    bl = labels[1:, :-1]    # bottom-left
    br = labels[1:, 1:]     # bottom-right

    # Sort the 4 values at each corner so we can count distinct labels easily
    quads = np.stack([tl, tr, bl, br], axis=-1)  # (H-1, W-1, 4)
    s = np.sort(quads, axis=-1)                   # sort within each corner

    # Distinct label count = 1 + number of positions where consecutive
    # sorted values differ.  Because full Voronoi has no zeros, all 4 values
    # are nonzero and this count is exact.
    n_uniq = 1 + np.sum(s[..., 1:] != s[..., :-1], axis=-1)  # (H-1, W-1)

    vertex_corner_mask = (n_uniq >= 3)
    edge_corner_flag   = (n_uniq == 2)

    # For edge corners: c1 = s[...,0] (smaller label), c2 = s[...,3] (larger)
    # Both are nonzero because of full Voronoi.
    c1_arr = s[..., 0].astype(np.int64)
    c2_arr = s[..., 3].astype(np.int64)

    edge_corner_map = np.zeros((H - 1, W - 1), dtype=np.int64)
    edge_corner_map[edge_corner_flag] = (
        c1_arr[edge_corner_flag] * K + c2_arr[edge_corner_flag]
    )

    return vertex_corner_mask, edge_corner_map


def _cluster_vertex_corners(
    vertex_corner_mask: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[Set[int]], Dict[Tuple[int, int], int]]:
    """
    Cluster connected vertex corners into actual vertex objects.

    A corner at grid position (r, c) represents image position (c+0.5, r+0.5).
    Connected components (8-connected) of the vertex corner mask are each
    collapsed to one vertex at the centroid of the component's positions.

    Cell-set signature splitting: within a connected component, if corners
    have substantially different 4-label sets, the component may contain two
    real twin junctions.  We use the same signature-grouping heuristic as
    the old pixel clusterer to split such components.

    Returns:
        vertices:         (N, 2) float — [x, y] in half-pixel coordinates.
        vertex_cell_sets: list of sets — cell labels meeting at each vertex.
        corner_to_vid:    dict (r, c) → vertex_id for all vertex corners.
    """
    H, W = labels.shape           # original image dimensions
    H_c, W_c = vertex_corner_mask.shape  # = (H-1, W-1)

    struct = np.ones((3, 3), dtype=int)
    comp_label, n_comp = nd_label(vertex_corner_mask, structure=struct)

    vertices: List[np.ndarray] = []
    vertex_cell_sets: List[Set[int]] = []
    corner_to_vid: Dict[Tuple[int, int], int] = {}

    # Group all component pixels in ONE pass instead of scanning the full
    # comp_label array once per component (previously O(n_comp * H * W)).
    fr, fc = np.where(comp_label > 0)
    cids = comp_label[fr, fc]
    order = np.argsort(cids, kind='stable')
    fr, fc, cids = fr[order], fc[order], cids[order]
    # nd_label produces contiguous ids 1..n_comp; split sorted pixels per id.
    split_points = np.searchsorted(cids, np.arange(2, n_comp + 1))
    cr_groups = np.split(fr, split_points)
    cc_groups = np.split(fc, split_points)

    for cid in range(1, n_comp + 1):
        cr, cc = cr_groups[cid - 1], cc_groups[cid - 1]

        # Collect the 4-label signature for each corner in this component
        pixel_sigs: List[frozenset] = []
        for r, c in zip(cr, cc):
            quad_cells: Set[int] = set()
            for dr in range(2):
                for dc in range(2):
                    rr, cc2 = r + dr, c + dc
                    if rr < H and cc2 < W:
                        lbl = int(labels[rr, cc2])
                        if lbl > 0:
                            quad_cells.add(lbl)
            pixel_sigs.append(frozenset(quad_cells))

        # Group by signature
        sig_to_indices: Dict[frozenset, List[int]] = {}
        for i, sig in enumerate(pixel_sigs):
            sig_to_indices.setdefault(sig, []).append(i)

        if len(sig_to_indices) == 1:
            # All corners share the same signature → single vertex
            sig = next(iter(sig_to_indices))
            centroid = np.array([np.mean(cc + 0.5), np.mean(cr + 0.5)])
            vid = len(vertices)
            vertices.append(centroid)
            vertex_cell_sets.append(set(sig))
            for r, c in zip(cr, cc):
                corner_to_vid[(int(r), int(c))] = vid
        else:
            # Multiple signatures — check for genuine twin junctions
            sigs_sorted = sorted(sig_to_indices.keys(),
                                 key=lambda s: -len(sig_to_indices[s]))
            dominant_sig = sigs_sorted[0]
            n_total = len(cr)

            sub_groups = []
            for sig in sigs_sorted:
                idxs = sig_to_indices[sig]
                frac = len(idxs) / n_total
                # Split only if: secondary group is ≥20% of component,
                # has a different cell set AND differs by at least one cell.
                if sig != dominant_sig:
                    is_different = len(sig.symmetric_difference(dominant_sig)) > 0
                    is_large_enough = frac >= 0.20
                    # Check spatial separation
                    r_dom = cr[sig_to_indices[dominant_sig]]
                    c_dom = cc[sig_to_indices[dominant_sig]]
                    r_sec = cr[idxs]
                    c_sec = cc[idxs]
                    dom_centroid = np.array([np.mean(c_dom + 0.5),
                                             np.mean(r_dom + 0.5)])
                    sec_centroid = np.array([np.mean(c_sec + 0.5),
                                             np.mean(r_sec + 0.5)])
                    spatial_sep = np.linalg.norm(dom_centroid - sec_centroid)
                    spatially_separate = spatial_sep > 2.0
                    if is_different and is_large_enough and spatially_separate:
                        sub_groups.append((sig, idxs))

            if not sub_groups:
                # No clear split — merge all into one vertex
                all_cells: Set[int] = set()
                for sig in sig_to_indices:
                    all_cells.update(sig)
                centroid = np.array([np.mean(cc + 0.5), np.mean(cr + 0.5)])
                vid = len(vertices)
                vertices.append(centroid)
                vertex_cell_sets.append(all_cells)
                for r, c in zip(cr, cc):
                    corner_to_vid[(int(r), int(c))] = vid
            else:
                # Assign dominant group to one vertex, each secondary to its own
                groups = [(dominant_sig, sig_to_indices[dominant_sig])]
                groups += sub_groups

                for sig, idxs in groups:
                    r_grp = cr[idxs]
                    c_grp = cc[idxs]
                    centroid = np.array([np.mean(c_grp + 0.5),
                                         np.mean(r_grp + 0.5)])
                    vid = len(vertices)
                    vertices.append(centroid)
                    vertex_cell_sets.append(set(sig))
                    for r, c in zip(r_grp, c_grp):
                        corner_to_vid[(int(r), int(c))] = vid

                # Assign any remaining corners to the nearest vertex
                assigned = {idx for _, idxs in groups for idx in idxs}
                for i in range(len(cr)):
                    if i not in assigned:
                        r, c = int(cr[i]), int(cc[i])
                        pos = np.array([c + 0.5, r + 0.5])
                        best_vid = min(
                            range(len(vertices)),
                            key=lambda v: np.linalg.norm(vertices[v] - pos)
                        )
                        corner_to_vid[(r, c)] = best_vid

    verts_arr = (np.array(vertices) if vertices
                 else np.zeros((0, 2), dtype=float))
    return verts_arr, vertex_cell_sets, corner_to_vid


def _build_boundary_pixel_map(
    labels: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Build a mapping from every adjacent cell pair to boundary pixel (x,y) coords.

    Scans the label image once (O(H×W)) to collect all 4-adjacent pixel pairs
    where the two pixels belong to different cells.  Far cheaper than calling
    a per-pair function inside a loop.

    Returns
    -------
    dict mapping (min_cell, max_cell) → float array of shape (N, 2) with
    (x, y) = (col + 0.5, row + 0.5) half-pixel image coordinates.
    """
    H, W = labels.shape
    pair_map: Dict[Tuple[int, int], List] = {}

    def _add(r1: int, c1: int, r2: int, c2: int) -> None:
        a, b = int(labels[r1, c1]), int(labels[r2, c2])
        if a == b or a == 0 or b == 0:
            return
        key = (min(a, b), max(a, b))
        if key not in pair_map:
            pair_map[key] = []
        # Both pixels straddle the boundary — include both positions
        pair_map[key].append((c1 + 0.5, r1 + 0.5))
        pair_map[key].append((c2 + 0.5, r2 + 0.5))

    # Horizontal neighbours
    hy, hx = np.where(labels[:, :-1] != labels[:, 1:])
    for r, c in zip(hy.tolist(), hx.tolist()):
        _add(r, c, r, c + 1)

    # Vertical neighbours
    vy, vx = np.where(labels[:-1, :] != labels[1:, :])
    for r, c in zip(vy.tolist(), vx.tolist()):
        _add(r, c, r + 1, c)

    return {k: np.array(v, dtype=float) for k, v in pair_map.items()}


def _boundary_pixels(
    labels: np.ndarray,
    c1: int,
    c2: int,
) -> Optional[np.ndarray]:
    """Return (x, y) pixel positions along the shared boundary of cells c1/c2.

    Thin wrapper kept for backward compatibility; builds a full map each call
    so it is *slow* if called in a loop — use _build_boundary_pixel_map instead.
    """
    key = (min(c1, c2), max(c1, c2))
    bmap = _build_boundary_pixel_map(labels)
    return bmap.get(key, None)


def _build_edges_from_corners(
    edge_corner_map: np.ndarray,
    vertex_corner_mask: np.ndarray,
    corner_to_vid: Dict[Tuple[int, int], int],
    vertices: np.ndarray,
    vertex_cell_sets: List[Set[int]],
    labels: np.ndarray,
    min_edge_len: int,
    n_original_vertices: int,
    fallback_radius: float = 40.0,
    recovery_radius: float = 40.0,
) -> Dict:
    """
    Build edge list from corner-classified boundary positions.

    Corner (r, c) corresponds to image position (x=c+0.5, y=r+0.5).

    For each cell pair (c1, c2):
      1. Collect all edge corners for that pair.
      2. Augment with vertex corners whose cell set contains both c1 and c2
         (so the component includes the junction at each endpoint).
      3. Find connected components of the augmented corner set.
      4. For each component: find which vertex corner IDs it touches.
         ≥2 vertices → real edge.  1 vertex (border) → virtual border vertex.
         1 vertex (interior) → fallback nearest-vertex search.

    Returns dict with keys: edges, cells, pixels, vertices, vertex_cell_sets.
    """
    H, W = labels.shape
    H_c, W_c = edge_corner_map.shape  # = (H-1, W-1)
    K = int(labels.max()) + 1

    unique_codes = np.unique(edge_corner_map[edge_corner_map > 0])

    edges_list:  List[List[int]] = []
    edges_cells: List[List[int]] = []
    edges_pixels: List[np.ndarray] = []

    struct = np.ones((3, 3), dtype=int)
    seen_keys: Set[Tuple[int, int, int, int]] = set()

    vid_to_cells = {vid: cset for vid, cset in enumerate(vertex_cell_sets)}

    # Pre-build set of vertex corners per cell pair to speed up augmentation
    # Maps code → set of (r,c) vertex corner positions
    pair_vertex_corners: Dict[int, List[Tuple[int, int]]] = {}
    for (r, c), vid in corner_to_vid.items():
        cset = vid_to_cells.get(vid, set())
        for cell_a in cset:
            for cell_b in cset:
                if cell_a < cell_b:
                    code = cell_a * K + cell_b
                    pair_vertex_corners.setdefault(code, []).append((r, c))

    # Pre-index edge corners by code in a SINGLE pass over the corner map.
    # Previously each code re-scanned, copied and connected-component-labelled
    # the full (H-1, W-1) corner map — O(n_codes * H * W). Since each cell pair
    # occupies only a tiny local region, we instead crop each code's work to the
    # bounding box of its corners, which is O(total boundary corners).
    code_to_edge_corners: Dict[int, List[Tuple[int, int]]] = {}
    er_all, ec_all = np.where(edge_corner_map > 0)
    for r, c, cd in zip(er_all.tolist(), ec_all.tolist(),
                        edge_corner_map[er_all, ec_all].tolist()):
        code_to_edge_corners.setdefault(int(cd), []).append((r, c))

    for code in unique_codes:
        c1 = int(code // K)
        c2 = int(code % K)

        edge_corner_pts = code_to_edge_corners.get(int(code), [])
        vtx_corner_pts = [
            (r, c) for (r, c) in pair_vertex_corners.get(int(code), [])
            if 0 <= r < H_c and 0 <= c < W_c
        ]
        all_pts = edge_corner_pts + vtx_corner_pts
        if not all_pts:
            continue

        # Bounding box of this pair's corners (+1 px margin, clipped to grid).
        rr = [p[0] for p in all_pts]
        cc = [p[1] for p in all_pts]
        r0 = max(0, min(rr) - 1)
        c0 = max(0, min(cc) - 1)
        r1 = min(H_c - 1, max(rr) + 1)
        c1b = min(W_c - 1, max(cc) + 1)
        sub_h = r1 - r0 + 1
        sub_w = c1b - c0 + 1

        # Local augmented mask (edge + vertex corners) and edge-only mask.
        augmented = np.zeros((sub_h, sub_w), dtype=bool)
        sub_pair = np.zeros((sub_h, sub_w), dtype=bool)
        for (r, c) in edge_corner_pts:
            augmented[r - r0, c - c0] = True
            sub_pair[r - r0, c - c0] = True
        for (r, c) in vtx_corner_pts:
            augmented[r - r0, c - c0] = True

        comp_label_c, n_comp = nd_label(augmented, structure=struct)

        for comp_id in range(1, n_comp + 1):
            comp_mask = (comp_label_c == comp_id)

            # Edge-only corners (not vertex corners), translated to global coords
            edge_only = comp_mask & sub_pair
            er_l, ec_l = np.where(edge_only)
            er = er_l + r0
            ec = ec_l + c0
            if len(er) < min_edge_len:
                continue

            # Convert corner positions to half-pixel image coordinates
            pts = np.column_stack(
                ((ec + 0.5).astype(float), (er + 0.5).astype(float))
            )

            # Find which vertex corners this component touches
            touching_vids: Set[int] = set()
            comp_r_l, comp_c_l = np.where(comp_mask)
            comp_r = comp_r_l + r0
            comp_c = comp_c_l + c0
            for r, c in zip(comp_r, comp_c):
                # Direct hit
                if (int(r), int(c)) in corner_to_vid:
                    touching_vids.add(corner_to_vid[(int(r), int(c))])
                else:
                    # Check 4-adjacent corners in the corner grid
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = int(r) + dr, int(c) + dc
                        if (nr, nc) in corner_to_vid:
                            touching_vids.add(corner_to_vid[(nr, nc)])

            touching_list = sorted(touching_vids)

            if len(touching_list) >= 2:
                v1, v2 = _pick_endpoint_vertices(touching_list, vertices, pts)
                key = (min(v1, v2), max(v1, v2), c1, c2)
                if key not in seen_keys:
                    seen_keys.add(key)
                    edges_list.append([v1, v2])
                    edges_cells.append([c1, c2])
                    edges_pixels.append(pts)

            elif len(touching_list) == 1:
                # Is this a border edge?
                border_touch = (
                    np.any(er == 0) or np.any(er == H_c - 1) or
                    np.any(ec == 0) or np.any(ec == W_c - 1)
                )
                if border_touch:
                    border_idx = np.concatenate([
                        np.where(er == 0)[0],
                        np.where(er == H_c - 1)[0],
                        np.where(ec == 0)[0],
                        np.where(ec == W_c - 1)[0],
                    ])
                    if len(border_idx) > 0:
                        bx = float(np.mean(ec[border_idx] + 0.5))
                        by = float(np.mean(er[border_idx] + 0.5))
                        new_v_idx = len(vertices)
                        vertices = np.vstack([vertices,
                                              np.array([[bx, by]])])
                        new_set: Set[int] = {c1, c2}
                        vertex_cell_sets.append(new_set)
                        vid_to_cells[new_v_idx] = new_set

                        v_int = touching_list[0]
                        key = (min(v_int, new_v_idx),
                               max(v_int, new_v_idx), c1, c2)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            edges_list.append([v_int, new_v_idx])
                            edges_cells.append([c1, c2])
                            edges_pixels.append(pts)
                else:
                    # Fallback: find nearest original vertex to far end
                    known_v = touching_list[0]
                    known_pos = vertices[known_v, :2]
                    comp_coords = np.column_stack(
                        ((comp_c + 0.5).astype(float),
                         (comp_r + 0.5).astype(float))
                    )
                    dists_from_known = np.linalg.norm(
                        comp_coords - known_pos, axis=1
                    )
                    far_coord = comp_coords[np.argmax(dists_from_known)]

                    if n_original_vertices > 0:
                        orig_coords = vertices[:n_original_vertices, :2]
                        dists_to_far = np.linalg.norm(
                            orig_coords - far_coord, axis=1
                        )
                        dists_to_far[known_v] = np.inf
                        nearest_v = int(np.argmin(dists_to_far))
                        if dists_to_far[nearest_v] <= fallback_radius:
                            key = (min(known_v, nearest_v),
                                   max(known_v, nearest_v), c1, c2)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                edges_list.append([known_v, nearest_v])
                                edges_cells.append([c1, c2])
                                # The corner-pixel set (pts) for a fallback
                                # edge is entirely clustered at known_v —
                                edges_pixels.append(pts)
                                logger.debug(
                                    f"Corner fallback: pair ({c1},{c2}) "
                                    f"v{known_v}→v{nearest_v} "
                                    f"({dists_to_far[nearest_v]:.1f} px)"
                                )

    # ---------------------------------------------------------------- #
    # Vertex-only edge recovery for tiny contacts                       #
    # ---------------------------------------------------------------- #
    # Some cell pairs touch only at a single pixel/corner contact. In
    # that case the corner classifier may never emit an EDGE corner for
    # the pair, so the main loop above never sees it. Recover these by
    # connecting the nearest two vertices that both contain the pair.
    pair_to_vids: Dict[Tuple[int, int], List[int]] = {}
    for vid in range(n_original_vertices):
        cells_v = sorted(vertex_cell_sets[vid])
        for ci in range(len(cells_v)):
            for cj in range(ci + 1, len(cells_v)):
                pair_to_vids.setdefault(
                    (cells_v[ci], cells_v[cj]), []
                ).append(vid)

    vertex_edge_pairs: Dict[int, Set[Tuple[int, int]]] = {
        vid: set() for vid in range(n_original_vertices)
    }
    for eidx, (v1, v2) in enumerate(edges_list):
        c_pair = (edges_cells[eidx][0], edges_cells[eidx][1])
        if v1 < n_original_vertices:
            vertex_edge_pairs[v1].add(c_pair)
        if v2 < n_original_vertices:
            vertex_edge_pairs[v2].add(c_pair)

    n_recovered = 0
    for vid in range(n_original_vertices):
        cells_v = sorted(vertex_cell_sets[vid])
        for ci in range(len(cells_v)):
            for cj in range(ci + 1, len(cells_v)):
                c1, c2 = cells_v[ci], cells_v[cj]
                if (c1, c2) in vertex_edge_pairs[vid]:
                    continue

                candidates = pair_to_vids.get((c1, c2), [])
                best_other = -1
                best_dist = float("inf")
                for other_vid in candidates:
                    if other_vid == vid:
                        continue
                    d = np.linalg.norm(
                        vertices[vid, :2] - vertices[other_vid, :2]
                    )
                    if d < best_dist:
                        best_dist = d
                        best_other = other_vid

                if best_other < 0 or best_dist > recovery_radius:
                    continue

                key = (min(vid, best_other), max(vid, best_other), c1, c2)
                if key in seen_keys:
                    continue

                seen_keys.add(key)
                edges_list.append([vid, best_other])
                edges_cells.append([c1, c2])
                edges_pixels.append(np.array([
                    [vertices[vid, 0], vertices[vid, 1]],
                    [vertices[best_other, 0], vertices[best_other, 1]],
                ]))
                n_recovered += 1

                vertex_edge_pairs[vid].add((c1, c2))
                if best_other < n_original_vertices:
                    vertex_edge_pairs[best_other].add((c1, c2))

    if n_recovered > 0:
        logger.info(
            f"Corner vertex-only recovery: {n_recovered} edges added "
            f"for pairs with no EDGE corners"
        )

    return {
        'edges': edges_list,
        'cells': edges_cells,
        'pixels': edges_pixels,
        'vertices': vertices,
        'vertex_cell_sets': vertex_cell_sets,
    }


# =============================================================================
# Phase 1:  Classify boundary pixels  (legacy — kept for reference)
# =============================================================================

def _classify_boundary_pixels(
    labels: np.ndarray,
    half_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every pixel on a cell boundary, determine if it is a VERTEX pixel
    (≥3 nonzero labels in window) or an EDGE pixel (exactly 2).

    Args:
        labels:      Label image (may be voronoi-expanded).
        half_window: Half-size of the analysis window.
                     1 → 3×3 (default, thin membranes)
                     2 → 5×5 (thick/blurry membranes)

    Returns:
        vertex_mask:  bool (H, W)
        edge_mask:    bool (H, W)
        edge_pair_map: int64 (H, W) — encodes (c1, c2) as c1*K + c2
    """
    H, W = labels.shape
    K = int(labels.max()) + 1
    hw = half_window  # shorthand

    # Pad labels to avoid boundary checks
    pad = np.pad(labels, hw, mode='constant', constant_values=0)

    # Build all offsets in the (2*hw+1)×(2*hw+1) window
    offsets = []
    for dy in range(-hw, hw + 1):
        for dx in range(-hw, hw + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))

    # Stacked shifted views
    shifted = []
    for dy, dx in offsets:
        shifted.append(pad[hw+dy:H+hw+dy, hw+dx:W+hw+dx])
    neighbours = np.stack(shifted, axis=-1)  # (H, W, n_offsets)

    center = labels
    any_diff = np.any(neighbours != center[..., None], axis=-1)

    all_window = np.concatenate(
        [neighbours, center[..., None]], axis=-1
    )

    vertex_mask = np.zeros((H, W), dtype=bool)
    edge_mask = np.zeros((H, W), dtype=bool)
    edge_pair_map = np.zeros((H, W), dtype=np.int64)

    by, bx = np.where(any_diff)
    if len(by) == 0:
        return vertex_mask, edge_mask, edge_pair_map

    windows = all_window[by, bx]
    sorted_w = np.sort(windows, axis=1)

    for i in range(len(by)):
        vals = sorted_w[i]
        nonzero = vals[vals > 0]
        if len(nonzero) == 0:
            continue
        unique_nonzero = np.unique(nonzero)
        n_unique = len(unique_nonzero)

        y, x = by[i], bx[i]
        if n_unique >= 3:
            vertex_mask[y, x] = True
        elif n_unique == 2:
            edge_mask[y, x] = True
            c1, c2 = int(unique_nonzero[0]), int(unique_nonzero[1])
            edge_pair_map[y, x] = c1 * K + c2

    return vertex_mask, edge_mask, edge_pair_map


# =============================================================================
# Phase 2:  Cluster vertex pixels
# =============================================================================

def _cluster_vertex_pixels(
    vertex_mask: np.ndarray,
    labels: np.ndarray,
    cluster_r: float,
) -> Tuple[np.ndarray, List[Set[int]]]:
    """
    Cluster vertex pixels into actual vertices.

    Uses connected components so that two spatially close but
    topologically distinct junctions (twin junctions) are NEVER merged.

    Within each connected component, we check the **cell-set signature**
    of each pixel.  If different pixels in the same component see
    different subsets of cells, this indicates multiple junctions
    within one blob (a true twin junction).  In that case we split
    by cell-set signature.  Otherwise the entire component → one vertex.

    This avoids the old problem where blind sub-clustering created
    spurious short "stub" edges within a single junction blob.

    Returns:
        vertices:        (N, 2) float — vertex positions [x, y]
        vertex_cell_sets: list of sets — cell labels meeting at each vertex
    """
    H, W = labels.shape
    struct = np.ones((3, 3), dtype=int)
    comp_label, n_comp = nd_label(vertex_mask, structure=struct)

    vertices = []
    vertex_cell_sets = []

    for cid in range(1, n_comp + 1):
        cy, cx = np.where(comp_label == cid)
        pts = np.column_stack((cx.astype(float), cy.astype(float)))

        # For each pixel in the component, compute its cell-set signature
        # (the set of nonzero cell labels in its 3×3 window).
        pixel_signatures = []
        for y, x in zip(cy, cx):
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            patch = labels[y0:y1, x0:x1]
            sig = frozenset(int(v) for v in np.unique(patch) if v > 0)
            pixel_signatures.append(sig)

        # Group pixels by signature
        sig_to_indices: Dict[frozenset, List[int]] = {}
        for i, sig in enumerate(pixel_signatures):
            sig_to_indices.setdefault(sig, []).append(i)

        # If all pixels have the same signature → one vertex
        if len(sig_to_indices) == 1:
            centroid = pts.mean(axis=0)
            all_cells = set()
            for sig in sig_to_indices:
                all_cells.update(sig)
            vertices.append(centroid)
            vertex_cell_sets.append(all_cells)
        else:
            # Multiple signatures → possible twin junction.
            # Group by signature, then check if groups are spatially
            # separable.  If two groups have substantially different
            # signatures AND are spatially distinct, they become
            # separate vertices.
            #
            # Heuristic: if two signature groups share all-but-one cell,
            # they are likely twin junctions at the endpoints of a short
            # edge (the "missing" cell is on one side of the short edge).
            # If they differ by more than one cell, they might just be
            # noise at the periphery of a wide junction blob.

            # Collect per-signature centroids
            sig_groups = []
            for sig, idxs in sig_to_indices.items():
                group_pts = pts[idxs]
                centroid = group_pts.mean(axis=0)
                sig_groups.append((sig, centroid, len(idxs)))

            # Sort by group size descending (largest group first)
            sig_groups.sort(key=lambda x: -x[2])

            # The dominant signature is the one with the most pixels.
            # Small peripheral groups (< 30% of total) are absorbed
            # into the dominant group.
            total_px = len(pts)
            dominant_sig, dominant_centroid, dominant_count = sig_groups[0]

            # Check if any secondary group is:
            #   (a) large enough (≥ 30% of total)
            #   (b) has a different cell set
            #   (c) spatially separated from dominant (> cluster_r)
            merged_cells = set(dominant_sig)
            split_groups = []

            for sig, centroid, count in sig_groups[1:]:
                frac = count / total_px
                dist = np.linalg.norm(centroid - dominant_centroid)
                sig_diff = sig.symmetric_difference(dominant_sig)

                # Split criterion: significant group, different topology,
                # and spatially separated
                if (frac >= 0.25 and len(sig_diff) >= 1
                        and dist > cluster_r):
                    split_groups.append((sig, centroid))
                else:
                    # Absorb into dominant group
                    merged_cells.update(sig)

            # Add the dominant (possibly merged) vertex
            all_merged_pts = pts[sig_to_indices[dominant_sig]]
            for sig, centroid, count in sig_groups[1:]:
                if not any(s == sig for s, _ in split_groups):
                    all_merged_pts = np.vstack([
                        all_merged_pts, pts[sig_to_indices[sig]]
                    ])
            vertices.append(all_merged_pts.mean(axis=0))
            vertex_cell_sets.append(merged_cells)

            # Add split groups as separate vertices
            for sig, centroid in split_groups:
                group_pts = pts[sig_to_indices[sig]]
                vertices.append(group_pts.mean(axis=0))
                vertex_cell_sets.append(set(sig))

    if not vertices:
        return np.zeros((0, 2), dtype=float), []
    return np.array(vertices), vertex_cell_sets


def _cluster_points(pts: np.ndarray, radius: float) -> np.ndarray:
    """Greedy merge: all points within `radius` → one centroid."""
    if len(pts) == 0:
        return pts
    tree = cKDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    merged = []
    for i in range(len(pts)):
        if visited[i]:
            continue
        nb = tree.query_ball_point(pts[i], radius)
        merged.append(pts[nb].mean(axis=0))
        visited[nb] = True
    return np.array(merged)


# =============================================================================
# Phase 2b:  Helpers for vertex-edge connectivity
# =============================================================================

def _build_vertex_pixel_map(
    vertex_mask: np.ndarray,
    labels: np.ndarray,
    vertices: np.ndarray,
    vertex_cell_sets: List[Set[int]],
) -> Dict[Tuple[int, int], int]:
    """
    Map each vertex pixel (y, x) to its vertex index.

    For pixels belonging to a connected component that was split into
    sub-clusters, assign to the nearest sub-cluster vertex.
    """
    vy, vx = np.where(vertex_mask)
    if len(vx) == 0:
        return {}

    pts = np.column_stack((vx.astype(float), vy.astype(float)))
    vtree = cKDTree(vertices)
    _, nearest = vtree.query(pts)

    pixel_map = {}
    for i in range(len(vx)):
        pixel_map[(int(vy[i]), int(vx[i]))] = int(nearest[i])
    return pixel_map


def _snap_to_skeleton(
    vertices: np.ndarray,
    skel: np.ndarray,
) -> np.ndarray:
    """Snap vertex positions to nearest skeleton pixel."""
    sy, sx = np.where(skel)
    if len(sx) == 0:
        return vertices
    skel_pts = np.column_stack((sx.astype(float), sy.astype(float)))
    tree = cKDTree(skel_pts)
    d, idx = tree.query(vertices)
    snapped = vertices.copy()
    close = d < 3.0  # only snap if within 3 px
    snapped[close] = skel_pts[idx[close]]
    return snapped


def _labels_to_boundary(labels: np.ndarray) -> np.ndarray:
    """Convert label image to binary boundary mask."""
    H, W = labels.shape
    b = np.zeros((H, W), dtype=bool)
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:,  :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:]  |= labels[:, :-1] != labels[:, 1:]
    return b


# =============================================================================
# Phase 3:  Build edges from edge pixels
# =============================================================================

def _build_edges_from_labels(
    edge_mask: np.ndarray,
    edge_pair_map: np.ndarray,
    vertex_mask: np.ndarray,
    vertex_pixel_map: Dict[Tuple[int, int], int],
    vertices: np.ndarray,
    vertex_cell_sets: List[Set[int]],
    labels: np.ndarray,
    min_edge_len: int,
) -> Dict:
    """
    Build edge list from classified boundary pixels.

    Returns dict with keys: edges, cells, pixels, vertices, vertex_cell_sets
    (vertices may grow due to border vertex additions).
    """
    H, W = labels.shape
    K = int(labels.max()) + 1

    # Collect unique cell pairs
    unique_codes = np.unique(edge_pair_map[edge_pair_map > 0])
    pair_dict: Dict[int, Tuple[int, int]] = {}
    for code in unique_codes:
        c1 = int(code // K)
        c2 = int(code % K)
        pair_dict[int(code)] = (c1, c2)

    edges_list: List[List[int]] = []
    edges_cells: List[List[int]] = []
    edges_pixels: List[np.ndarray] = []

    struct = np.ones((3, 3), dtype=int)

    seen_edge_keys: Set[Tuple[int, int, int, int]] = set()

    # Remember the count of original (non-border) vertices so the fallback
    # nearest-vertex search only considers real junctions, not the virtual
    # border vertices that get appended during this loop.
    n_original_vertices = len(vertices)

    # Pre-compute which vertex pixels belong to which cell pairs
    # to speed up the augmentation step
    vy_all, vx_all = np.where(vertex_mask)
    vpx_cell_pairs: Dict[Tuple[int, int], Set[int]] = {}
    for idx_v, (y, x) in enumerate(zip(vy_all, vx_all)):
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        patch = labels[y0:y1, x0:x1]
        patch_labels = set(int(v) for v in np.unique(patch) if v > 0)
        vpx_cell_pairs[(y, x)] = patch_labels

    for code, (c1, c2) in pair_dict.items():
        # Mask of edge pixels for this cell pair
        pair_mask = (edge_pair_map == code)
        # Also include vertex pixels that belong to both c1 and c2
        # This helps connect edge components to their vertex endpoints
        augmented = pair_mask.copy()
        for (y, x), plabels in vpx_cell_pairs.items():
            if c1 in plabels and c2 in plabels:
                augmented[y, x] = True

        comp_label, n_comp = nd_label(augmented, structure=struct)

        for comp_id in range(1, n_comp + 1):
            comp_mask = (comp_label == comp_id)

            # Separate the actual edge pixels from vertex pixels
            edge_only = comp_mask & pair_mask
            ey, ex = np.where(edge_only)

            # Allow zero-edge-pixel components: when two junctions are
            # directly adjacent (common for very small cells), all boundary
            # pixels between them are classified as VERTEX pixels and no pure
            # EDGE pixels remain.  We still want to create the edge.
            # Only reject components that have *some* pixels but fewer than
            # min_edge_len (1-pixel noise between unrelated junctions).
            if 0 < len(ey) < min_edge_len:
                continue

            pts = (np.column_stack((ex.astype(float), ey.astype(float)))
                   if len(ey) > 0 else np.zeros((0, 2), dtype=float))

            # Find which vertices this component touches.
            # We search a 2-pixel (5×5) radius rather than just 8-neighbors
            # because there can be a 2-px gap between the last edge pixel
            # and the nearest vertex pixel at thick-membrane junctions.
            # A 1-px gap was already bridged by 8-connectivity in nd_label;
            # a 2-px gap requires this wider search.
            touching_vertices = set()
            comp_y, comp_x = np.where(comp_mask)
            for y, x in zip(comp_y, comp_x):
                if (y, x) in vertex_pixel_map:
                    touching_vertices.add(vertex_pixel_map[(y, x)])
                else:
                    # Check 2-pixel radius (5×5 window) for vertex pixels
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < H and 0 <= nx < W and
                                    (ny, nx) in vertex_pixel_map):
                                touching_vertices.add(
                                    vertex_pixel_map[(ny, nx)]
                                )

            touching_list = sorted(touching_vertices)

            if len(touching_list) >= 2:
                v1, v2 = _pick_endpoint_vertices(
                    touching_list, vertices, pts
                )
                key = (min(v1, v2), max(v1, v2), c1, c2)
                if key not in seen_edge_keys:
                    seen_edge_keys.add(key)
                    edges_list.append([v1, v2])
                    edges_cells.append([c1, c2])
                    # For zero-pixel edges (directly-adjacent junctions in
                    # tiny cells), create a minimal 2-point path so that
                    # downstream code that indexes E_pixels always gets an
                    # array with at least 2 rows.
                    if len(pts) == 0:
                        pts = np.array([
                            [vertices[v1, 0], vertices[v1, 1]],
                            [vertices[v2, 0], vertices[v2, 1]],
                        ])
                    edges_pixels.append(pts)

            elif len(touching_list) == 1:
                # One vertex found.  Either a border edge (one end at the
                # image frame) or a missed junction (the 2px radius search
                # still didn't reach the second vertex).
                border_touch = (
                    np.any(ey == 0) or np.any(ey == H - 1) or
                    np.any(ex == 0) or np.any(ex == W - 1)
                )
                if border_touch:
                    # ── border edge: create a virtual vertex at the boundary ──
                    border_pixel_mask = np.zeros((H, W), dtype=bool)
                    border_pixel_mask[0, :] = border_pixel_mask[-1, :] = True
                    border_pixel_mask[:, 0] = border_pixel_mask[:, -1] = True
                    border_pts = comp_mask & border_pixel_mask
                    bpy, bpx = np.where(border_pts)
                    if len(bpx) > 0:
                        border_pos = np.array([
                            float(np.mean(bpx)), float(np.mean(bpy))
                        ])
                        new_v_idx = len(vertices)
                        vertices = np.vstack([
                            vertices, border_pos.reshape(1, 2)
                        ])
                        vertex_cell_sets.append({c1, c2})

                        v_int = touching_list[0]
                        key = (min(v_int, new_v_idx),
                               max(v_int, new_v_idx), c1, c2)
                        if key not in seen_edge_keys:
                            seen_edge_keys.add(key)
                            edges_list.append([v_int, new_v_idx])
                            edges_cells.append([c1, c2])
                            edges_pixels.append(pts)

                else:
                    # ── missed junction fallback ──────────────────────────────
                    # The 2-px adjacency search still only found 1 vertex.
                    # This happens when the far end of the membrane is
                    # separated from the nearest vertex cluster by > 2 px
                    # (e.g., very thick membranes or low-resolution junctions).
                    #
                    # Strategy: find the component pixel farthest from the
                    # known vertex ("the far end"), then pick the nearest
                    # original vertex within FALLBACK_RADIUS pixels as the
                    # second endpoint.  This recovers the missing edge without
                    # fabricating geometry.
                    FALLBACK_RADIUS = 12  # px; enough for thick membranes
                    known_v = touching_list[0]
                    known_pos = vertices[known_v, :2]

                    comp_coords = np.column_stack(
                        (comp_x.astype(float), comp_y.astype(float))
                    )
                    dists_from_known = np.linalg.norm(
                        comp_coords - known_pos, axis=1
                    )
                    far_coord = comp_coords[np.argmax(dists_from_known)]

                    # Search only among original (non-border) vertices
                    if n_original_vertices > 0:
                        orig_coords = vertices[:n_original_vertices, :2]
                        dists_to_far = np.linalg.norm(
                            orig_coords - far_coord, axis=1
                        )
                        dists_to_far[known_v] = np.inf  # exclude self
                        nearest_v = int(np.argmin(dists_to_far))
                        nearest_dist = float(dists_to_far[nearest_v])

                        if nearest_dist <= FALLBACK_RADIUS:
                            key = (min(known_v, nearest_v),
                                   max(known_v, nearest_v), c1, c2)
                            if key not in seen_edge_keys:
                                seen_edge_keys.add(key)
                                edges_list.append([known_v, nearest_v])
                                edges_cells.append([c1, c2])
                                # pts holds only boundary pixels near
                                # known_v; the second vertex was found by
                                edges_pixels.append(pts)
                                logger.debug(
                                    f"Fallback edge recovery: pair ({c1},{c2})"
                                    f" v{known_v}→v{nearest_v} "
                                    f"(gap {nearest_dist:.1f} px)"
                                )

    # ---------------------------------------------------------------- #
    # Vertex-only edge recovery for small cells                         #
    # ---------------------------------------------------------------- #
    # Some cell pairs share only VERTEX pixels (no EDGE pixels at all). #
    # This means the pair never appeared in edge_pair_map and was never  #
    # iterated in the main loop above.  Common for very short membranes #
    # in small cells where the 3×3 window always sees 3+ cell labels.   #
    #                                                                    #
    # Strategy: for each interior vertex that is "underfull" (has fewer  #
    # edges than expected cell pairs), find the nearest vertex that      #
    # shares the missing cell pair and create a direct edge.             #
    # ---------------------------------------------------------------- #

    # Build lookup: cell pair → list of vertex IDs that contain both
    pair_to_vids: Dict[Tuple[int, int], List[int]] = {}
    for vid in range(n_original_vertices):
        cells_v = sorted(vertex_cell_sets[vid])
        for ci in range(len(cells_v)):
            for cj in range(ci + 1, len(cells_v)):
                pair_to_vids.setdefault(
                    (cells_v[ci], cells_v[cj]), []
                ).append(vid)

    # Track which cell pairs each vertex already has edges for
    vertex_edge_pairs: Dict[int, Set[Tuple[int, int]]] = {
        vid: set() for vid in range(n_original_vertices)
    }
    for eidx in range(len(edges_list)):
        v1, v2 = edges_list[eidx]
        c1c2 = (edges_cells[eidx][0], edges_cells[eidx][1])
        if v1 < n_original_vertices:
            vertex_edge_pairs[v1].add(c1c2)
        if v2 < n_original_vertices:
            vertex_edge_pairs[v2].add(c1c2)

    RECOVERY_RADIUS = 15.0  # max distance for vertex-only edge recovery

    n_recovered = 0
    for vid in range(n_original_vertices):
        cells_v = sorted(vertex_cell_sets[vid])
        for ci in range(len(cells_v)):
            for cj in range(ci + 1, len(cells_v)):
                c1, c2 = cells_v[ci], cells_v[cj]
                if (c1, c2) in vertex_edge_pairs[vid]:
                    continue  # already has an edge for this pair

                # Find nearest other vertex that also contains (c1, c2)
                candidates = pair_to_vids.get((c1, c2), [])
                best_other = -1
                best_dist = float('inf')
                for other_vid in candidates:
                    if other_vid == vid:
                        continue
                    d = np.linalg.norm(
                        vertices[vid, :2] - vertices[other_vid, :2]
                    )
                    if d < best_dist:
                        best_dist = d
                        best_other = other_vid

                if best_other < 0 or best_dist > RECOVERY_RADIUS:
                    continue

                key = (min(vid, best_other), max(vid, best_other), c1, c2)
                if key in seen_edge_keys:
                    continue

                seen_edge_keys.add(key)
                edges_list.append([vid, best_other])
                edges_cells.append([c1, c2])
                pts_recov = np.array([
                    [vertices[vid, 0], vertices[vid, 1]],
                    [vertices[best_other, 0], vertices[best_other, 1]],
                ])
                edges_pixels.append(pts_recov)
                n_recovered += 1

                # Update tracking so the other vertex also knows
                vertex_edge_pairs[vid].add((c1, c2))
                if best_other < n_original_vertices:
                    vertex_edge_pairs[best_other].add((c1, c2))

    if n_recovered > 0:
        logger.info(
            f"Vertex-only edge recovery: {n_recovered} edges added "
            f"for cell pairs with no EDGE pixels"
        )

    return {
        'edges': edges_list,
        'cells': edges_cells,
        'pixels': edges_pixels,
        'vertices': vertices,
        'vertex_cell_sets': vertex_cell_sets,
    }


def _pick_endpoint_vertices(
    candidates: List[int],
    vertices: np.ndarray,
    edge_pts: np.ndarray,
) -> Tuple[int, int]:
    """
    Given a list of candidate vertex indices and edge pixels, pick
    the two vertices that best represent the two ends of the edge.
    """
    if len(candidates) == 2:
        return candidates[0], candidates[1]

    # No edge pixels (directly-adjacent junctions): just pick any two
    if len(edge_pts) == 0:
        return candidates[0], candidates[1]

    # Principal axis of edge pixels
    center = np.mean(edge_pts, axis=0)
    centered = edge_pts - center
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        direction = Vt[0]
    except np.linalg.LinAlgError:
        direction = np.array([1.0, 0.0])

    # Project edge pixels
    projs = edge_pts @ direction
    min_pt = edge_pts[np.argmin(projs)]
    max_pt = edge_pts[np.argmax(projs)]

    # Pick vertices closest to each extreme
    cand_coords = vertices[candidates]
    d_min = np.linalg.norm(cand_coords - min_pt, axis=1)
    d_max = np.linalg.norm(cand_coords - max_pt, axis=1)

    v1 = candidates[int(np.argmin(d_min))]
    v2 = candidates[int(np.argmin(d_max))]

    if v1 == v2:
        # Pick the second-best for the max end
        order = np.argsort(d_max)
        for idx in order:
            if candidates[idx] != v1:
                v2 = candidates[idx]
                break

    return v1, v2


# =============================================================================
# Phase 4:  Pixel ordering and geometry
# =============================================================================

def _order_pixels(
    pts: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Order edge pixels from p1 to p2."""
    if len(pts) < 2:
        return pts

    edge_vec = p2 - p1
    length = np.linalg.norm(edge_vec)

    if length > 1e-6:
        d = edge_vec / length
        perp = np.abs((pts - p1) @ np.array([-d[1], d[0]]))
        if float(np.percentile(perp, 90)) < 3.0:
            return pts[np.argsort((pts - p1) @ d)]

    # Greedy nearest-neighbour
    d1 = np.linalg.norm(pts - p1, axis=1)
    start = int(np.argmin(d1))
    ordered = [start]
    remaining = set(range(len(pts)))
    remaining.discard(start)
    while remaining:
        curr = pts[ordered[-1]]
        rem_list = list(remaining)
        dists = np.linalg.norm(pts[rem_list] - curr, axis=1)
        nearest = rem_list[int(np.argmin(dists))]
        ordered.append(nearest)
        remaining.discard(nearest)
    return pts[ordered]


def _resample_spline(
    pts: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Resample edge pixels to n_points via spline interpolation."""
    try:
        k = min(3, len(pts) - 1)
        tck, u = splprep(
            [pts[:, 0], pts[:, 1]],
            s=len(pts) * 0.5,
            k=k,
        )
        u_new = np.linspace(0, 1, n_points)
        sx, sy = splev(u_new, tck)
        result = np.column_stack([sx, sy])
        result[0] = v1
        result[-1] = v2
        return result
    except Exception:
        return pts


# =============================================================================
# Phase 5:  Build C_v (per-cell vertex loops)
# =============================================================================

def _build_C_v(
    labels: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    E_cells: np.ndarray,
) -> List[List[int]]:
    """Build per-cell vertex loop C_v[lbl-1] in CCW order."""
    max_lbl = int(labels.max())
    C_v: List[List[int]] = [[] for _ in range(max_lbl)]
    cell_verts: List[Set[int]] = [set() for _ in range(max_lbl)]

    for e_idx, (v1, v2) in enumerate(E):
        for c in E_cells[e_idx]:
            if 0 < c <= max_lbl:
                cell_verts[c - 1].add(int(v1))
                cell_verts[c - 1].add(int(v2))

    props = measure.regionprops(labels)
    centroids = {p.label: np.array([p.centroid[1], p.centroid[0]])
                 for p in props if p.label > 0}

    for lbl in range(1, max_lbl + 1):
        verts = sorted(cell_verts[lbl - 1])
        if len(verts) < 2:
            continue
        if lbl not in centroids:
            C_v[lbl - 1] = verts
            continue
        cx, cy = centroids[lbl]
        angles = [np.arctan2(V[v, 1] - cy, V[v, 0] - cx) for v in verts]
        C_v[lbl - 1] = [verts[i] for i in np.argsort(angles)]

    return C_v


# =============================================================================
# Stub collapse — degree-aware short-edge removal
# =============================================================================

def _collapse_stub_edges(tissue: Tissue, threshold: float) -> Tissue:
    """
    Remove short stub edges that are artifacts of wide junction blobs.

    A stub edge is defined as a short edge (length < threshold) where
    at least one endpoint has graph-degree ≤ 2.  Such edges are NOT
    real cell-cell interfaces; they are artifacts of sub-clustering
    within a single junction blob.

    A TRUE twin-junction short edge has degree ≥ 3 at BOTH endpoints
    (because each endpoint is a real triple-point where ≥3 edges meet).
    Those are preserved.

    Algorithm:
      1. Compute vertex degrees
      2. Find short edges where min(deg[v1], deg[v2]) ≤ 2
      3. Merge the low-degree vertex into the high-degree vertex
      4. Re-wire all edges that used the low-degree vertex
      5. Remove self-loops and duplicate edges
      6. Rebuild C_v
    """
    V = tissue.V.copy()
    E = tissue.E.copy()
    E_cells = tissue.E_cells.copy()
    E_pixels = list(tissue.E_pixels) if tissue.E_pixels else None

    if len(E) == 0:
        return tissue

    for iteration in range(10):  # iterate until stable
        n_v = len(V)
        # Compute degrees
        degree = np.zeros(n_v, dtype=int)
        for v1, v2 in E:
            degree[v1] += 1
            degree[v2] += 1

        # Find stub edges: short + at least one low-degree endpoint
        edge_lengths = np.linalg.norm(V[E[:, 0], :2] - V[E[:, 1], :2], axis=1)

        # Build merge map: vertex → vertex it gets merged into
        # We merge the LOW-degree end into the HIGH-degree end.
        merge_map = np.arange(n_v)  # identity initially
        stubs_found = False

        for eidx in range(len(E)):
            if edge_lengths[eidx] >= threshold:
                continue
            v1, v2 = E[eidx]
            d1, d2 = degree[v1], degree[v2]

            # Both high-degree → real twin junction, keep it
            if d1 >= 3 and d2 >= 3:
                continue

            # At least one is low-degree → stub, collapse it
            stubs_found = True

            # Merge the lower-degree vertex into the higher-degree one.
            # If equal degree, merge into the one with lower index.
            if d1 <= d2:
                merge_map[v1] = v2
            else:
                merge_map[v2] = v1

        if not stubs_found:
            break

        # Chase merge chains: if v1 → v2 → v3, then v1 → v3
        for i in range(n_v):
            while merge_map[i] != merge_map[merge_map[i]]:
                merge_map[i] = merge_map[merge_map[i]]

        # Compact vertex indices
        kept = np.unique(merge_map)
        new_idx = np.full(n_v, -1, dtype=int)
        new_idx[kept] = np.arange(len(kept))

        # Remap vertices
        new_V = V[kept]

        # Remap edges
        new_E_list = []
        new_E_cells_list = []
        new_E_pixels_list = []
        seen = set()

        for eidx in range(len(E)):
            v1, v2 = E[eidx]
            nv1 = new_idx[merge_map[v1]]
            nv2 = new_idx[merge_map[v2]]
            if nv1 == nv2:
                continue  # self-loop after merge
            key = (min(nv1, nv2), max(nv1, nv2))
            if key in seen:
                continue  # duplicate edge
            seen.add(key)
            new_E_list.append([nv1, nv2])
            new_E_cells_list.append(E_cells[eidx].tolist())
            if E_pixels is not None:
                new_E_pixels_list.append(E_pixels[eidx])

        if not new_E_list:
            break

        V = new_V
        E = np.array(new_E_list, dtype=int)
        E_cells = np.array(new_E_cells_list, dtype=int)
        if E_pixels is not None:
            E_pixels = new_E_pixels_list

    # Rebuild tissue
    # Re-sort: inner vertices first, border last
    H, W = tissue.labels.shape
    margin = 2.0
    is_border = (
        (V[:, 0] <= margin) | (V[:, 0] >= W - margin) |
        (V[:, 1] <= margin) | (V[:, 1] >= H - margin)
    )
    inner_idx = np.where(~is_border)[0]
    border_idx = np.where(is_border)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner = len(inner_idx)

    V = V[sorted_idx]
    old_to_new = np.full(len(sorted_idx), -1, dtype=int)
    old_to_new[sorted_idx] = np.arange(len(sorted_idx))
    E = old_to_new[E]

    tissue.V = V
    tissue.E = E
    tissue.E_cells = E_cells
    tissue.num_inner_vertices = num_inner
    if E_pixels is not None:
        tissue.E_pixels = E_pixels

    # Rebuild C_v
    tissue.C_v = _build_C_v(tissue.labels, V, E, E_cells)

    return tissue


# =============================================================================
# Below-resolution twin-junction collapse → 4-way vertex
# =============================================================================

def _collapse_tiny_twin_junctions(tissue: Tissue, threshold: float) -> Tissue:
    """
    Promote below-resolution twin junctions to 4-way vertices.

    After stub removal, any remaining short edge where BOTH endpoints have
    graph-degree ≥ 3 is a genuine twin junction whose separation is below
    the image resolution.  At that scale the distinction between
    "two 3-way junctions 1px apart" and "one 4-way junction" is physically
    meaningless, so we merge them into a single 4-way vertex placed at
    their midpoint.

    This is the correct topology for the scale — NOT an approximation.

    Unlike _collapse_stub_edges (which requires min(deg) ≤ 2), here we
    target short edges where BOTH degrees ≥ 3.

    Algorithm:
      1. Compute vertex degrees.
      2. Find short edges (length < threshold) where deg(v1) ≥ 3
         AND deg(v2) ≥ 3.
      3. For each such edge, set merge_map[higher_idx] = lower_idx
         (merge into the vertex with lower index; position = midpoint).
      4. Chase merge chains, compact vertices, re-wire edges.
      5. Remove self-loops and duplicate edges.
      6. Rebuild C_v.
      Iterate until stable (≤ 10 passes).
    """
    V = tissue.V.copy()
    E = tissue.E.copy()
    E_cells = tissue.E_cells.copy()
    E_pixels = list(tissue.E_pixels) if tissue.E_pixels else None

    if len(E) == 0:
        return tissue

    for iteration in range(10):  # iterate until stable
        n_v = len(V)

        # Compute degrees
        degree = np.zeros(n_v, dtype=int)
        for v1, v2 in E:
            degree[v1] += 1
            degree[v2] += 1

        # Edge lengths (Euclidean on x,y)
        edge_lengths = np.linalg.norm(V[E[:, 0], :2] - V[E[:, 1], :2], axis=1)

        # Build merge map
        merge_map = np.arange(n_v)
        twins_found = False

        # Track midpoint corrections: when two vertices merge, place the
        # surviving vertex at the midpoint of the two originals.
        midpoints = V[:, :2].copy()  # will accumulate averages

        for eidx in range(len(E)):
            if edge_lengths[eidx] >= threshold:
                continue
            v1, v2 = E[eidx]
            d1, d2 = degree[v1], degree[v2]

            # Only handle pairs where BOTH are high-degree (real twin junctions).
            # Low-degree cases are handled by _collapse_stub_edges.
            if d1 < 3 or d2 < 3:
                continue

            twins_found = True

            # Merge higher-index into lower-index (stable canonical choice).
            keep, drop = (v1, v2) if v1 <= v2 else (v2, v1)
            merge_map[drop] = keep
            # Place surviving vertex at midpoint of the pair.
            midpoints[keep] = (midpoints[keep] + midpoints[drop]) / 2.0

        if not twins_found:
            break

        # Chase merge chains: v1 → v2 → v3 becomes v1 → v3
        for i in range(n_v):
            while merge_map[i] != merge_map[merge_map[i]]:
                merge_map[i] = merge_map[merge_map[i]]

        # Update surviving vertex positions to midpoints
        for i in range(n_v):
            root = merge_map[i]
            if root != i:
                # Update the root's position if not already done via midpoints
                pass
        # Apply midpoint updates to V
        new_V_positions = midpoints  # already updated in the loop above
        # For survivors that were never a 'keep', position is unchanged.

        # Compact vertex indices
        kept = np.unique(merge_map)
        new_idx = np.full(n_v, -1, dtype=int)
        new_idx[kept] = np.arange(len(kept))

        # Build new vertex array with midpoint-corrected positions
        new_V = np.zeros((len(kept), V.shape[1]))
        for new_i, old_i in enumerate(kept):
            new_V[new_i, :2] = new_V_positions[old_i]
            if V.shape[1] > 2:
                new_V[new_i, 2:] = V[old_i, 2:]

        # Re-wire edges
        new_E_list = []
        new_E_cells_list = []
        new_E_pixels_list = []
        seen = set()

        for eidx in range(len(E)):
            v1, v2 = E[eidx]
            nv1 = new_idx[merge_map[v1]]
            nv2 = new_idx[merge_map[v2]]
            if nv1 == nv2:
                continue  # self-loop (the collapsed twin-junction edge itself)
            key = (min(nv1, nv2), max(nv1, nv2))
            if key in seen:
                continue  # duplicate
            seen.add(key)
            new_E_list.append([nv1, nv2])
            new_E_cells_list.append(E_cells[eidx].tolist())
            if E_pixels is not None:
                new_E_pixels_list.append(E_pixels[eidx])

        if not new_E_list:
            break

        V = new_V
        E = np.array(new_E_list, dtype=int)
        E_cells = np.array(new_E_cells_list, dtype=int)
        if E_pixels is not None:
            E_pixels = new_E_pixels_list

    # Rebuild tissue — re-sort inner/border
    H, W = tissue.labels.shape
    margin = 2.0
    is_border = (
        (V[:, 0] <= margin) | (V[:, 0] >= W - margin) |
        (V[:, 1] <= margin) | (V[:, 1] >= H - margin)
    )
    inner_idx = np.where(~is_border)[0]
    border_idx = np.where(is_border)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner = len(inner_idx)

    V = V[sorted_idx]
    old_to_new = np.full(len(sorted_idx), -1, dtype=int)
    old_to_new[sorted_idx] = np.arange(len(sorted_idx))
    E = old_to_new[E]

    tissue.V = V
    tissue.E = E
    tissue.E_cells = E_cells
    tissue.num_inner_vertices = num_inner
    if E_pixels is not None:
        tissue.E_pixels = E_pixels

    # Rebuild C_v
    tissue.C_v = _build_C_v(tissue.labels, V, E, E_cells)

    return tissue


# =============================================================================
# Graph cleaning (same as skeleton-based version)
# =============================================================================

def _clean_tissue_graph(tissue: Tissue, min_edge_len: float) -> Tissue:
    """Collapse edges shorter than min_edge_len by merging vertices."""
    V, E = tissue.V, tissue.E
    E_cells = tissue.E_cells
    E_pixels = getattr(tissue, 'E_pixels', None)
    C_v = tissue.C_v

    if len(E) == 0:
        return tissue

    for _ in range(5):
        if len(E) == 0:
            break
        d = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        short = np.where(d < min_edge_len)[0]
        if len(short) == 0:
            break

        parent = np.arange(len(V))
        for idx in short:
            s, t = E[idx]
            r = min(parent[s], parent[t])
            parent[s] = parent[t] = r
        for i in range(len(V)):
            while parent[i] != parent[parent[i]]:
                parent[i] = parent[parent[i]]

        keep = np.unique(parent)
        mapping = {old: new for new, old in enumerate(keep)}

        new_V = np.zeros((len(keep), V.shape[1]))
        counts = np.zeros(len(keep))
        for i in range(len(V)):
            ni = mapping[parent[i]]
            new_V[ni] += V[i]
            counts[ni] += 1
        new_V /= counts[:, None]

        edge_to_data = {}
        for e_idx, (s, t) in enumerate(E):
            ns = mapping[parent[s]]
            nt = mapping[parent[t]]
            if ns != nt:
                key = (min(ns, nt), max(ns, nt))
                if key not in edge_to_data:
                    edge_to_data[key] = {
                        'cells': set(),
                        'pixels': E_pixels[e_idx] if E_pixels else None,
                    }
                if E_cells is not None:
                    for c in np.atleast_1d(E_cells[e_idx]):
                        edge_to_data[key]['cells'].add(int(c))

        new_keys = sorted(edge_to_data.keys())
        new_E = (np.array(new_keys, dtype=int) if new_keys
                 else np.zeros((0, 2), dtype=int))

        new_E_cells, new_pixels = [], []
        for key in new_keys:
            cells = sorted(edge_to_data[key]['cells'])
            if len(cells) == 0:
                new_E_cells.append([0, 0])
            elif len(cells) == 1:
                new_E_cells.append([cells[0], 0])
            else:
                new_E_cells.append(cells[:2])
            if E_pixels is not None:
                new_pixels.append(edge_to_data[key]['pixels'])

        E_cells = (np.array(new_E_cells, dtype=int) if new_E_cells
                   else np.zeros((0, 2), dtype=int))

        if C_v is not None:
            remapped = []
            for seq in C_v:
                new_seq, last_v = [], None
                for vi in seq:
                    if vi < 0 or vi >= len(parent):
                        continue
                    nv = mapping[parent[vi]]
                    if last_v is None or nv != last_v:
                        new_seq.append(nv)
                        last_v = nv
                if len(new_seq) > 1 and new_seq[0] == new_seq[-1]:
                    new_seq.pop()
                remapped.append(new_seq)
            C_v = remapped

        V, E = new_V, new_E
        if E_pixels is not None:
            E_pixels = new_pixels
        if len(E) == 0:
            break

    tissue.V, tissue.E = V, E
    tissue.E_cells = E_cells
    tissue.C_v = C_v
    if E_pixels is not None:
        tissue.E_pixels = E_pixels
    return tissue
