# segmentation2.py
# Based on the original segmentation.py provided.
# Modifications: Wrapped main logic in run_segmentation function,
#                modified export_labelme_by_level to return created file paths.

#%%
# colour-PCA hierarchical merge (2025-04)
# Using original segmentation parameters, but new level-based JSON export.
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage import io, segmentation, color, filters, graph, measure, img_as_float, img_as_ubyte # Added img_as_ubyte
from skimage.feature import canny
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
# from shapely.affinity import rotate # Not currently used
from shapely.ops import unary_union
from scipy.ndimage import binary_dilation
import base64
import json
import os
from itertools import combinations
from collections import defaultdict
from pathlib import Path
import logging
import traceback # Added for better error reporting if needed

# Configure logging if not already configured by the caller
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─────────── Default Segmentation Parameters (can be overridden) ────────────
DEFAULT_N_SEGMENTS = 250 # SLIC initial segments
DEFAULT_COMPACTNESS = 3 # SLIC compactness

DEFAULT_ALPHA, DEFAULT_BETA = 4.0, 12.0 # angle ‖ ΔLab cost weights
DEFAULT_GAMMA = 3.0 # gradient penalty weight
DEFAULT_ETA = 0.5 # length penalty weight (ratio of PCA lengths)
DEFAULT_L0 = 0.03 # threshold for "long" PCA axis in Lab space
# DEFAULT_THRESH will be passed as argument
# ───────────────────────────────────────────────────────────────────────

# ─────────── Default Polygon/JSON Generation Parameters ────────────────
DEFAULT_SIMPLIFY_PX = 0.1 # Polygon simplification tolerance
DEFAULT_MIN_RING_AREA = 25 # px² – drop initial contour rings smaller than this
DEFAULT_MIN_REGION_AREA = 50 # px² – drop final regions *after* punching out children

DEFAULT_N_DIRS = 3 # how many direction chords per region
DEFAULT_MIN_FRAC = .25 # min length of chord relative to max possible length
DEFAULT_MAX_FRAC = .45 # max length of chord relative to max possible length
DEFAULT_FAN_DEG = 75 # angular spread (± deg) around PCA axis for chords
# ───────────────────────────────────────────────────────────────────────


#%%
# ————————————————————————————————————————————————————————————————
# 1) Contours → valid exterior polygons (one per super-pixel label)
# (Using the robust version from later code)
# ————————————————————————————————————————————————————————————————
# Using passed simplify_px and min_ring_area now
def polygons_from_labels(label_img, simplify_px: float, min_ring_area: float):
    """
    Yields (region_id, shapely.Polygon) for each label in the image.
    Handles simplification and basic area filtering. Returns valid Polygon objects.
    Uses simplify_px and min_ring_area passed as arguments.
    """
    processed_ids = set()
    unique_labels = np.unique(label_img)
    logging.info(f"Processing {len(unique_labels)} unique labels found in image.")
    for rid in unique_labels:
        if rid < 0:
            logging.debug(f"Skipping region ID {rid} (negative).")
            continue # Skip background label if present (<0)

        process_label_zero = True # Keep processing label 0 unless specifically excluded
        if rid == 0 and not process_label_zero:
             logging.info("Skipping region ID 0 based on configuration.")
             continue

        if rid in processed_ids: continue # Should not happen with unique_labels, but safe check

        mask = label_img == rid
        if not np.any(mask):
            logging.debug(f"Skipping region ID {rid} (empty mask).")
            continue

        try:
            padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_mask.astype(float), 0.5)
            contours = [c - 1 for c in contours]
        except Exception as e:
            logging.error(f"find_contours failed for region {rid}: {e}")
            continue

        if not contours:
            logging.debug(f"No contours found for region ID {rid}.")
            continue

        rings = []
        for cnt in contours:
            if len(cnt) < 4:
                 logging.debug(f"Skipping short contour ({len(cnt)} points) for region {rid}.")
                 continue
            try:
                poly = Polygon(np.fliplr(cnt)).buffer(0)
            except Exception as e:
                logging.warning(f"Polygon creation/buffer failed for a contour in region {rid}: {e}")
                continue

            if poly.is_empty or not poly.is_valid:
                 logging.debug(f"Skipping empty or invalid raw polygon for region {rid}.")
                 continue
            if poly.area < min_ring_area: # Use argument here
                logging.debug(f"Skipping contour for region {rid} due to small area ({poly.area:.2f} < {min_ring_area}).")
                continue
            rings.append(poly)

        if not rings:
             logging.debug(f"No valid rings survived for region ID {rid} after initial filtering.")
             continue

        try:
            merged_geom = unary_union(rings).buffer(0)
        except Exception as e:
            logging.error(f"Unary union failed for region {rid}: {e}")
            continue

        geoms_to_process = []
        if isinstance(merged_geom, Polygon):
            geoms_to_process.append(merged_geom)
        elif isinstance(merged_geom, MultiPolygon):
            geoms_to_process.extend(list(merged_geom.geoms))
        elif not merged_geom.is_empty:
            logging.warning(f"Unexpected geometry type {type(merged_geom)} after union for region {rid}. Skipping.")
            continue
        else:
             logging.debug(f"Region {rid} resulted in empty geometry after union.")
             continue

        processed_ids.add(rid)
        for idx, geom in enumerate(geoms_to_process):
            if geom.is_empty or not geom.is_valid:
                logging.debug(f"Skipping empty/invalid geometry part {idx} for region {rid}.")
                continue

            try:
                simplified_geom = geom.simplify(simplify_px, preserve_topology=True).buffer(0) # Use argument here
            except Exception as e:
                 logging.warning(f"Simplification/buffer failed for geometry part {idx} in region {rid}: {e}")
                 continue

            if not simplified_geom.is_empty and simplified_geom.is_valid and simplified_geom.area >= min_ring_area: # Use argument here
                logging.debug(f"Yielding polygon for region {rid} (part {idx}) with area {simplified_geom.area:.2f}.")
                yield rid, simplified_geom
            else:
                 logging.debug(f"Skipping geometry part {idx} for region {rid} after simplification (Area: {simplified_geom.area:.2f}, Valid: {simplified_geom.is_valid}).")


class Region:
    __slots__ = ("id", "orig_geom", "final_geom", "children", "parent", "depth")
    def __init__(self, rid: int, geom: Polygon):
        self.id = rid; self.orig_geom = geom; self.final_geom = geom
        self.children = []; self.parent = None; self.depth = -1
    def __iter__(self): yield self; yield from (d for c in self.children for d in c)
    def __repr__(self): return f"Region(id={self.id}, depth={self.depth}, children={[c.id for c in self.children]})"

def build_region_forest(rid2poly: dict[int, Polygon]) -> tuple[list[Region], int]:
    """Builds the containment hierarchy (forest) from polygons."""
    nodes = {rid: Region(rid, g) for rid, g in rid2poly.items()}
    logging.info(f"Building forest for {len(nodes)} regions.")
    if not nodes: return [], -1

    CONTAINS_BUFFER = 1e-6
    for r1, r2 in combinations(nodes.values(), 2):
        if not r1.orig_geom.is_valid: r1.orig_geom = r1.orig_geom.buffer(0)
        if not r2.orig_geom.is_valid: r2.orig_geom = r2.orig_geom.buffer(0)
        if not r1.orig_geom.is_valid or not r2.orig_geom.is_valid:
            logging.warning(f"Skipping invalid geometry during contains check for region {r1.id} or {r2.id}")
            continue

        try:
            contains_r2 = r1.orig_geom.area > r2.orig_geom.area and r1.orig_geom.buffer(CONTAINS_BUFFER).contains(r2.orig_geom)
            contains_r1 = r2.orig_geom.area > r1.orig_geom.area and r2.orig_geom.buffer(CONTAINS_BUFFER).contains(r1.orig_geom)
        except Exception as e:
            logging.error(f"Contains check failed between region {r1.id} and {r2.id}: {e}", exc_info=False)
            continue

        if contains_r2:
             if r2.parent is None or r2.parent.orig_geom.contains(r1.orig_geom):
                 if r1.parent is None or r1.parent.id != r2.id:
                      if r2.parent is not None:
                          try: r2.parent.children.remove(r2)
                          except ValueError: pass
                      r1.children.append(r2)
                      r2.parent = r1
        elif contains_r1:
             if r1.parent is None or r1.parent.orig_geom.contains(r2.orig_geom):
                 if r2.parent is None or r2.parent.id != r1.id:
                      if r1.parent is not None:
                          try: r1.parent.children.remove(r1)
                          except ValueError: pass
                      r2.children.append(r1)
                      r1.parent = r2

    roots = [n for n in nodes.values() if n.parent is None]
    processed_depths = set()
    queue = [(root, 0) for root in roots]
    max_calculated_depth = -1
    visited_in_queue = {r.id for r in roots}

    while queue:
        node, d = queue.pop(0)
        if node.id in processed_depths: continue
        node.depth = d
        max_calculated_depth = max(max_calculated_depth, d)
        processed_depths.add(node.id)
        for ch in node.children:
            if ch.id not in visited_in_queue:
                 queue.append((ch, d + 1))
                 visited_in_queue.add(ch.id)

    if len(processed_depths) != len(nodes):
        logging.warning(f"Depth calculation issue: Processed {len(processed_depths)} depths, but expected {len(nodes)} nodes.")
        nodes_without_depth = [nid for nid in nodes if nodes[nid].depth == -1]
        logging.warning(f"Nodes without assigned depth: {nodes_without_depth}")
        for nid in nodes_without_depth:
             if nodes[nid].depth == -1:
                  nodes[nid].depth = 0
                  if nodes[nid].parent is None and nodes[nid] not in roots: roots.append(nodes[nid])
                  logging.warning(f"Assigned depth 0 to orphan node {nid}.")
        max_calculated_depth = max(max_calculated_depth, 0)

    logging.info(f"Final root count: {len(roots)}. Max calculated depth: {max_calculated_depth}")
    return roots, max_calculated_depth

# Using passed n_dirs, min_frac, max_frac, fan_deg now
def inner_chords(poly: Polygon, forbidden_polys: list[Polygon], rng: np.random.Generator, n_dirs: int, min_frac: float, max_frac: float, fan_deg: float):
    """Generates inner chords based on passed parameters."""
    if poly.is_empty or not poly.is_valid or poly.area < 1e-6: return []
    current_poly = poly if poly.is_valid else poly.buffer(0)
    if not current_poly.is_valid or current_poly.is_empty:
        logging.warning(f"Skipping inner_chords for invalid/empty polygon (Area: {poly.area:.2f})")
        return []

    try:
        repr_point = current_poly.representative_point()
        Cx, Cy = repr_point.x, repr_point.y
    except Exception as e:
        logging.error(f"Could not get representative point: {e}. Using centroid.")
        try:
            Cx, Cy = current_poly.centroid.x, current_poly.centroid.y
        except Exception as ce:
             logging.error(f"Could not get centroid either: {ce}. Cannot generate chords.")
             return []

    try:
        xy = np.asarray(current_poly.exterior.coords) - (Cx, Cy)
        if xy.shape[0] <= 2: return []
        _, _, Vt = np.linalg.svd(xy[:-1], full_matrices=False)
        axis = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-9)
    except Exception as e:
        logging.warning(f"SVD failed for PCA axis: {e}. Using random axis.")
        axis = rng.random(2) * 2 - 1
        axis = axis / (np.linalg.norm(axis) + 1e-9)

    segs, tries = [], 0
    max_tries = n_dirs * 20

    while len(segs) < n_dirs and tries < max_tries:
        tries += 1
        angle_rad = np.deg2rad(rng.uniform(-fan_deg, fan_deg)) # Use argument
        rot_matrix = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])
        v = rot_matrix @ axis

        bounds = current_poly.bounds
        if not (len(bounds)==4 and all(isinstance(b,float) for b in bounds)): continue
        diag_len = np.sqrt((bounds[2]-bounds[0])**2 + (bounds[3]-bounds[1])**2)
        far = diag_len*1.5 + 10
        line = LineString([(Cx - far * v[0], Cy - far * v[1]), (Cx + far * v[0], Cy + far * v[1])])

        try:
            chord = line.intersection(current_poly)
        except Exception as e:
             logging.debug(f"Line intersection failed: {e}")
             continue

        if chord.is_empty or not isinstance(chord, LineString) or len(chord.coords)<2: continue

        c0, c1 = map(np.asarray, chord.coords)
        full_len = np.linalg.norm(c1 - c0)
        if full_len < 1e-3: continue

        min_len = min_frac * full_len # Use argument
        max_len = max_frac * full_len # Use argument
        if max_len <= min_len or min_len < 0: continue

        target_len = rng.uniform(min_len, max_len)
        start_offset = rng.uniform(0, full_len - target_len) if full_len > target_len else 0
        end_offset = start_offset + target_len
        unit_vec = (c1 - c0) / (full_len + 1e-9)

        pad = min(0.5, target_len * 0.01, full_len * 0.005)
        p0 = c0 + unit_vec * (start_offset + pad)
        p1 = c0 + unit_vec * (end_offset - pad)

        if np.linalg.norm(p1 - p0) < 1e-3: continue

        seg = LineString([tuple(p0), tuple(p1)])

        try:
            is_contained = current_poly.buffer(-1e-6).contains(seg)
            intersects_forbidden = False
            if is_contained:
                for f_poly in forbidden_polys:
                    valid_f_poly = f_poly if f_poly.is_valid else f_poly.buffer(0)
                    if not valid_f_poly.is_empty and seg.intersects(valid_f_poly):
                        intersects_forbidden = True
                        break
            if is_contained and not intersects_forbidden:
                segs.append([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]])
        except Exception as e:
             logging.debug(f"Final segment check failed: {e}")
             continue

    return segs

# Using passed min_region_area, n_dirs, min_frac, max_frac, fan_deg now
# MODIFIED: Returns list of created file paths
def export_labelme_by_level(img_path_obj: Path, H: int, W: int, roots: list[Region], calculated_max_depth: int,
                            min_region_area: float, n_dirs: int, min_frac: float, max_frac: float, fan_deg: float):
    """Exports LabelMe JSON files per level and returns list of created paths."""
    img_path_str = str(img_path_obj); img_basename = img_path_obj.name; out_dir = img_path_obj.parent; stem = img_path_obj.stem
    all_nodes = [node for r in roots for node in r]
    if not all_nodes: logging.warning("No regions found for JSON export."); return [] # Return empty list

    level2nodes = defaultdict(list); logging.info("Punching out children and filtering by MIN_REGION_AREA..."); nodes_removed_count = 0
    final_nodes_for_export = []

    for n in all_nodes:
        punched_geom = n.orig_geom
        if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)
        if not punched_geom.is_valid or punched_geom.is_empty:
             logging.warning(f"Region {n.id} has invalid/empty original geometry. Skipping.")
             nodes_removed_count += 1
             n.final_geom = None
             continue

        if n.children:
            valid_child_geoms = []
            for c in n.children:
                child_geom = c.orig_geom
                if not child_geom.is_valid: child_geom = child_geom.buffer(0)
                if child_geom.is_valid and not child_geom.is_empty:
                    valid_child_geoms.append(child_geom)

            if valid_child_geoms:
                try:
                    kids_union = unary_union(valid_child_geoms)
                    if not kids_union.is_valid: kids_union = kids_union.buffer(0)
                    if kids_union.is_valid and not kids_union.is_empty:
                        punched_geom = punched_geom.difference(kids_union)
                        if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)
                except Exception as e:
                    logging.error(f"Punch out failed for region {n.id}: {e}. Using original geometry.")
                    punched_geom = n.orig_geom # Fallback
                    if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)

        # Use argument min_region_area here
        if punched_geom.is_empty or not punched_geom.is_valid or punched_geom.area < min_region_area:
            logging.info(f"Region {n.id} removed after punching/filtering (Area: {punched_geom.area:.2f} < {min_region_area} or invalid/empty).")
            nodes_removed_count += 1
            n.final_geom = None
            continue
        else:
             n.final_geom = punched_geom
             if n.depth >= 0:
                final_nodes_for_export.append(n)

    if nodes_removed_count > 0: logging.info(f"Total {nodes_removed_count} regions removed before export due to area/validity constraints.")
    if not final_nodes_for_export: logging.warning("No regions survived filtering for JSON export."); return [] # Return empty list

    for n in final_nodes_for_export: level2nodes[n.depth].append(n)

    img_b64 = None
    num_exported_levels = 0
    created_files = [] # <<< List to store created file paths
    logging.info(f"Exporting JSON levels 0 to {calculated_max_depth} for surviving regions.")

    for lvl in range(calculated_max_depth + 1):
        nodes_at_level = level2nodes.get(lvl, [])
        if not nodes_at_level:
            logging.debug(f"No regions to export at level {lvl}.")
            continue

        shapes = []
        sibling_geoms = {n.id: n.final_geom for n in nodes_at_level if n.final_geom}
        num_polys_at_level = 0

        for n in nodes_at_level:
            if n.final_geom is None or n.final_geom.is_empty or not n.final_geom.is_valid: continue
            num_polys_at_level += 1
            current_poly = n.final_geom
            geoms_to_add = []

            if isinstance(current_poly, Polygon): geoms_to_add.append(current_poly)
            elif isinstance(current_poly, MultiPolygon):
                 # Use argument min_region_area here
                geoms_to_add.extend(p for p in current_poly.geoms if p.area >= min_region_area and p.is_valid)

            for poly_part in geoms_to_add:
                if poly_part.is_empty or not poly_part.is_valid or poly_part.exterior is None or len(poly_part.exterior.coords) < 4:
                     logging.warning(f"Skipping invalid/small polygon part for region {n.id} at level {lvl}.")
                     continue

                pts = np.asarray(poly_part.exterior.coords[:-1]).tolist()
                shapes.append({"label":f"region-{n.id}", "points":pts, "group_id":None, "shape_type":"polygon", "flags":{}})

                rng = np.random.default_rng(seed=n.id + lvl)
                forbidden = [g for k, g in sibling_geoms.items() if k != n.id and g] \
                          + [d.orig_geom for d in n if d.id != n.id and d.orig_geom.is_valid and not d.orig_geom.is_empty]

                # Use arguments n_dirs, min_frac, max_frac, fan_deg here
                for seg in inner_chords(poly_part, forbidden, rng, n_dirs, min_frac, max_frac, fan_deg):
                    shapes.append({"label":f"direction-{n.id}", "points":seg, "group_id":None, "shape_type":"line", "flags":{}})

        if not shapes:
             logging.info(f"No shapes generated for level {lvl}, skipping JSON file.")
             continue

        if img_b64 is None:
            logging.info("Reading image data for embedding...")
            try: img_b64 = base64.b64encode(img_path_obj.read_bytes()).decode("utf-8")
            except Exception as e: logging.error(f"Failed to read/encode image file {img_path_obj}: {e}"); img_b64 = ""

        lm_data = {"version":"5.2.1", "flags":{}, "shapes":shapes, "imagePath":img_basename, "imageData":img_b64, "imageHeight":H, "imageWidth":W}
        out_path = out_dir / f"{stem}_labelme_L{lvl}.json"
        try:
            out_path.write_text(json.dumps(lm_data, indent=2))
            logging.info(f"✓ Wrote {out_path.name} ({num_polys_at_level} polygons, {len(shapes)} total shapes)")
            created_files.append(out_path) # <<< Append path to list
            num_exported_levels += 1
        except Exception as e:
            logging.error(f"Failed to write JSON file {out_path}: {e}")

    if num_exported_levels == 0: logging.warning("No JSON files were exported for any level.")
    else: logging.info(f"Exported {num_exported_levels} JSON level file(s).")

    return created_files # <<< Return the list of created paths


#%%
# —————————————————— Segmentation Helpers ————————————————————

def pca_axis(lab_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the mean and endpoints along the first principal component axis."""
    if lab_pixels.shape[0] < 2:
        mu = lab_pixels.mean(0) if lab_pixels.shape[0] > 0 else np.zeros(3)
        return mu, mu
    μ = lab_pixels.mean(0)
    try:
        _, S, Vt = np.linalg.svd(lab_pixels - μ, full_matrices=False)
        v1 = Vt[0]
        σ1 = S[0] / np.sqrt(max(1, lab_pixels.shape[0] - 1))
        return μ - v1 * σ1, μ + v1 * σ1
    except np.linalg.LinAlgError:
        logging.warning("SVD failed in pca_axis, returning mean twice.")
        return μ, μ

# Using passed alpha, beta, l0 now
def angle_penalty(v1_norm, v2_norm, ℓ1, ℓ2, alpha, l0):
    """Calculates penalty based on angle between PCA axes and their lengths."""
    if ℓ1 < 1e-6 or ℓ2 < 1e-6: return 1.0
    cos_theta = abs(v1_norm @ v2_norm)
    theta = np.arccos(np.clip(cos_theta, 0, 1))
    ℓm = max(ℓ1, ℓ2); ℓmi = min(ℓ1, ℓ2)
    if ℓm < l0: return np.exp(-alpha * theta * theta) # Use argument alpha, l0
    elif ℓmi < l0: return np.exp(-0.5 * alpha * theta * theta) # Use argument alpha, l0
    return 1.0

# Global dictionaries needed by callbacks
# These will be populated within run_segmentation
_LMEAN = {}
_PCA = {}
_GRAD = {}
_lab_sum = {}
_lab_npix = {}

# Need access to lbl, lab in callback - make them available via run_segmentation context if possible
# For now, relying on them being populated within the function call. This is less clean.
# A class-based approach for segmentation might be better but deviates more.
_current_lbl = None
_current_lab = None

# Using passed alpha, beta, gamma, eta, l0 now
def affinity(u, v, alpha, beta, gamma, eta, l0):
    """Calculates the affinity (similarity) between two regions u and v."""
    if u not in _PCA or v not in _PCA or u not in _LMEAN or v not in _LMEAN: return 0.0
    p1, p2 = _PCA[u]; q1, q2 = _PCA[v]
    vec1 = p2 - p1; vec2 = q2 - q1
    l1 = np.linalg.norm(vec1); l2 = np.linalg.norm(vec2)
    angleFac, lengthFac = 1.0, 1.0
    if l1 > 1e-6 and l2 > 1e-6:
        v1_norm = vec1 / l1; v2_norm = vec2 / l2
        angleFac = angle_penalty(v1_norm, v2_norm, l1, l2, alpha, l0) # Use arguments
        lmin, lmax = (l1, l2) if l1 < l2 else (l2, l1)
        lengthFac = np.exp(-eta * lmin / (lmax + 1e-9)) # Use argument eta
    dLab = np.linalg.norm(_LMEAN[u] - _LMEAN[v])
    colFac = np.exp(-beta * dLab * dLab) # Use argument beta
    key = tuple(sorted((u, v)))
    g = _GRAD.get(key, 0)
    gradFac = np.exp(-gamma * g * g) # Use argument gamma
    return angleFac * lengthFac * colFac * gradFac

# Using passed alpha, beta, gamma, eta, l0 now
def cost(u, v, alpha, beta, gamma, eta, l0):
    """Calculates merge cost (lower affinity -> higher cost)."""
    aff = affinity(u, v, alpha, beta, gamma, eta, l0) # Pass arguments
    return -np.log(aff + 1e-12)

# --- Callbacks ---
# These now need access to the parameters (alpha, beta, etc.)
# We can use functools.partial in run_segmentation to bind them.

def merge_cb(rag, src, dst):
    """Callback function executed when merging src into dst."""
    # Uses global dicts populated by run_segmentation
    if src not in _lab_sum or dst not in _lab_sum or src not in _lab_npix or dst not in _lab_npix:
        logging.warning(f"Missing feature data during merge: src={src}, dst={dst}")
        return

    _lab_sum[dst] += _lab_sum[src]
    _lab_npix[dst] += _lab_npix[src]
    _LMEAN[dst] = _lab_sum[dst] / (_lab_npix[dst] + 1e-9)

    global _current_lbl, _current_lab # Access globals set by run_segmentation
    try:
        if _current_lbl is None or _current_lab is None:
            raise NameError("Global '_current_lbl' or '_current_lab' not available for merge_cb PCA update.")

        pts_indices = np.where(_current_lbl == dst)
        pts = _current_lab[pts_indices]

        if pts.shape[0] > 3000:
            pts = pts[np.random.choice(pts.shape[0], 3000, replace=False)]

        if pts.shape[0] >= 2: _PCA[dst] = pca_axis(pts)
        elif dst in _PCA: del _PCA[dst]
    except NameError as e:
        logging.error(f"Stopping merge due to missing global: {e}")
        raise
    except Exception as e:
        logging.error(f"PCA update error for region {dst} during merge: {e}")
        if dst in _PCA: del _PCA[dst]

    if src in _lab_sum: del _lab_sum[src]
    if src in _lab_npix: del _lab_npix[src]
    if src in _LMEAN: del _LMEAN[src]
    if src in _PCA: del _PCA[src]

def weight_cb(rag, src, dst, nbr, alpha, beta, gamma, eta, l0):
    """Callback to calculate edge weight between merged node (dst) and its neighbor (nbr)."""
    # Uses global dicts populated by run_segmentation
    if dst not in _LMEAN or nbr not in _LMEAN or dst not in _PCA or nbr not in _PCA:
         return {'weight': np.inf}

    # Calculate the new cost using the bound parameters
    new_cost = cost(dst, nbr, alpha, beta, gamma, eta, l0)
    return {'weight': new_cost if np.isfinite(new_cost) else np.inf}


# --- Main Function Wrapper ---
def run_segmentation(
    image_path_str: str,
    threshold: float,
    output_dir: Path,
    n_segments: int = DEFAULT_N_SEGMENTS,
    compactness: float = DEFAULT_COMPACTNESS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    eta: float = DEFAULT_ETA,
    l0: float = DEFAULT_L0,
    simplify_px: float = DEFAULT_SIMPLIFY_PX,
    min_ring_area: float = DEFAULT_MIN_RING_AREA,
    min_region_area: float = DEFAULT_MIN_REGION_AREA,
    n_dirs: int = DEFAULT_N_DIRS,
    min_frac: float = DEFAULT_MIN_FRAC,
    max_frac: float = DEFAULT_MAX_FRAC,
    fan_deg: float = DEFAULT_FAN_DEG,
    save_plots: bool = True # Control plotting
    ):
    """
    Runs the complete segmentation process and JSON export.

    Args:
        image_path_str: Path to the input image.
        threshold: The merging threshold (THRESH).
        output_dir: The directory to save outputs (JSONs, plots).
        n_segments, compactness, alpha, beta, gamma, eta, l0: Segmentation parameters.
        simplify_px, min_ring_area, min_region_area: Polygon generation parameters.
        n_dirs, min_frac, max_frac, fan_deg: Chord generation parameters.
        save_plots: Whether to save segmentation/preview plots.

    Returns:
        tuple: (Path object for image, Height, Width, list of generated JSON paths)
               Returns (None, 0, 0, []) on failure.
    """
    global _LMEAN, _PCA, _GRAD, _lab_sum, _lab_npix, _current_lbl, _current_lab # Use prefixed globals

    IMG_PATH = Path(image_path_str)
    DATA_DIR = output_dir
    DATA_DIR.mkdir(exist_ok=True) # Ensure output directory exists

    logging.info(f"Running segmentation for: {IMG_PATH} with threshold={threshold}")
    logging.info(f"Output directory set to: {DATA_DIR}")
    logging.info(f"Params: n_seg={n_segments}, compact={compactness}, alpha={alpha}, beta={beta}, gamma={gamma}, eta={eta}, l0={l0}")
    logging.info(f"Poly Params: simplify={simplify_px}, min_ring_A={min_ring_area}, min_region_A={min_region_area}")
    logging.info(f"Chord Params: n_dirs={n_dirs}, frac=[{min_frac},{max_frac}], fan={fan_deg}")


    # Reset global feature dicts
    _LMEAN.clear(); _PCA.clear(); _GRAD.clear(); _lab_sum.clear(); _lab_npix.clear()
    _current_lbl = None # Reset global label map
    _current_lab = None # Reset global lab image

    # --- Start Segmentation Execution ---
    try:
        logging.info(f"Reading image: {IMG_PATH}")
        rgb = io.imread(IMG_PATH)
        if rgb.ndim == 2: rgb = color.gray2rgb(rgb)
        if rgb.shape[2] == 4: rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8: rgb = img_as_ubyte(rgb)

        H, W = rgb.shape[:2]
        logging.info(f"Image dimensions: Height={H}, Width={W}")

        lab_raw = color.rgb2lab(rgb).astype(float)
        _current_lab = np.zeros_like(lab_raw) # Store in global for callbacks
        _current_lab[..., 0] = (lab_raw[..., 0] - 50) / 50
        _current_lab[..., 1:] = lab_raw[..., 1:] / 110

        rgb_float = img_as_float(rgb)
        sigma_blur = 1.0
        if sigma_blur > 0: rgb_blurred = filters.gaussian(rgb_float, sigma=sigma_blur, channel_axis=-1, preserve_range=True)
        else: rgb_blurred = rgb_float

        logging.info(f"Running SLIC: n_segments={n_segments}, compactness={compactness}...")
        _current_lbl = segmentation.slic(rgb_blurred, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1, enforce_connectivity=True)
        _current_lbl = segmentation.relabel_sequential(_current_lbl)[0]
        num_slic_regions = _current_lbl.max() + 1
        logging.info(f"SLIC generated {num_slic_regions} initial regions.")

        logging.info("Calculating initial features...")
        unique_slic_labels = np.unique(_current_lbl)
        for l in unique_slic_labels:
            if l < 0: continue
            pix_mask = _current_lbl == l
            if not np.any(pix_mask): continue
            pix_lab = _current_lab[pix_mask]
            if pix_lab.shape[0] == 0: continue
            _LMEAN[l] = pix_lab.mean(0)
            _lab_sum[l] = pix_lab.sum(0)
            _lab_npix[l] = pix_lab.shape[0]
            if _lab_npix[l] >= 2: _PCA[l] = pca_axis(pix_lab)
        logging.info(f"Calculated initial features for {len(_LMEAN)} regions.")

        logging.info("Calculating boundary gradients...")
        gray = color.rgb2gray(rgb_float)
        canny_sigma = 1.0
        Gmag = canny(gray, sigma=canny_sigma).astype(float)

        rag_initial = graph.rag_mean_color(rgb_float, _current_lbl, mode='distance')
        edge_count = 0
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
        for u, v in rag_initial.edges:
            if u not in _LMEAN or v not in _LMEAN: continue
            mask_u = _current_lbl == u; mask_v = _current_lbl == v
            boundary_v = binary_dilation(mask_u, structure=structure, border_value=0) & mask_v
            coords = np.column_stack(np.where(boundary_v))
            mean_grad = 0.0
            if coords.size > 0:
                coords = coords[(coords[:, 0] < H) & (coords[:, 1] < W)]
                if coords.size > 0: mean_grad = Gmag[coords[:, 0], coords[:, 1]].mean()
            _GRAD[tuple(sorted((u, v)))] = mean_grad
            edge_count += 1
        logging.info(f"Processed gradients for {edge_count} edges.")

        logging.info("Building RAG with custom costs...")
        rag = rag_initial.copy()
        edges_removed_count = 0; edges_updated_count = 0
        for u, v, edge_data in rag.edges(data=True):
             # Use parameters passed to run_segmentation
            edge_cost = cost(u, v, alpha, beta, gamma, eta, l0) if u in _LMEAN and v in _LMEAN and u in _PCA and v in _PCA else np.inf
            if np.isfinite(edge_cost): edge_data['weight'] = edge_cost; edges_updated_count += 1
            else: edge_data['weight'] = np.inf; edges_removed_count += 1
        logging.info(f"Custom RAG prepared: {rag.number_of_nodes()} nodes, {rag.number_of_edges()} edges. Removed/Inf: {edges_removed_count}")

        # --- Prepare callbacks with bound parameters ---
        import functools
        # merge_cb doesn't need params bound as it uses globals implicitly (less ideal but matches original structure)
        # weight_cb needs the cost parameters bound to it
        bound_weight_cb = functools.partial(weight_cb, alpha=alpha, beta=beta, gamma=gamma, eta=eta, l0=l0)

        logging.info(f"Starting hierarchical merging with Thresh={threshold}...")
        # merge_hierarchical modifies _current_lbl and rag in place
        lbl2 = graph.merge_hierarchical(
            _current_lbl, # Initial label map (will be modified)
            rag,
            thresh=threshold, # Use argument
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_cb, # Uses globals
            weight_func=bound_weight_cb # Uses bound cost parameters
        )
        num_final_regions = len(np.unique(lbl2))
        logging.info(f"Hierarchical merging complete: {num_slic_regions} initial regions -> {num_final_regions} final regions.")

        # --- Optional Plotting ---
        if save_plots:
            try:
                logging.info("Generating segmentation visualization plot...")
                fig_seg, ax_seg = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
                ax_seg[0].imshow(segmentation.mark_boundaries(rgb_float, segmentation.relabel_sequential(_current_lbl)[0], color=(1, 0, 0), mode='thick')) # Use original slic result if needed
                ax_seg[0].set_title(f'SLIC ({num_slic_regions} regions)')
                ax_seg[0].axis('off')
                ax_seg[1].imshow(segmentation.mark_boundaries(rgb_float, lbl2, color=(0, 1, 0), mode='thick'))
                ax_seg[1].set_title(f'After Merge ({num_final_regions} regions, Thresh={threshold})')
                ax_seg[1].axis('off')
                plt.tight_layout()
                plot_path = DATA_DIR / f"{IMG_PATH.stem}_segmentation_result.png" # Simplified name
                plt.savefig(plot_path)
                logging.info(f"Segmentation visualization saved to {plot_path}")
                plt.close(fig_seg) # Close the figure
            except Exception as e:
                logging.error(f"Failed to generate or save segmentation plot: {e}")

    except FileNotFoundError:
        logging.error(f"Image file not found at {IMG_PATH}. Please check the path.")
        return None, 0, 0, []
    except Exception as e:
        logging.error(f"Critical error during segmentation: {e}")
        logging.error(traceback.format_exc())
        return None, 0, 0, []
    # --- End Segmentation Execution ---


    # =====================================================================
    # Polygon Generation and Level-Based JSON Export using final lbl2
    # =====================================================================
    if lbl2 is None: # Check if merging completed successfully
        logging.error("Final label map 'lbl2' is not available. Cannot proceed.")
        return IMG_PATH, H, W, []

    logging.info("Generating polygons from final merged labels...")
    rid2poly_initial = {}
    try:
        # Pass polygon parameters
        rid2poly_initial = dict(polygons_from_labels(lbl2, simplify_px=simplify_px, min_ring_area=min_ring_area))
        num_initial_polys = len(rid2poly_initial)
        if num_initial_polys == 0:
             logging.warning("No polygons were generated from the labels. Cannot proceed with JSON export.")
             return IMG_PATH, H, W, []
    except Exception as e:
        logging.error(f"Error during polygon generation: {e}")
        logging.error(traceback.format_exc())
        return IMG_PATH, H, W, []

    logging.info("Building region containment forest...")
    try:
        roots, max_depth_calculated = build_region_forest(rid2poly_initial)
    except Exception as e:
         logging.error(f"Error building region forest: {e}")
         logging.error(traceback.format_exc())
         return IMG_PATH, H, W, []

    logging.info("Exporting LabelMe JSON files per depth level...")
    created_json_paths = []
    try:
        # Pass polygon and chord parameters to export function
        created_json_paths = export_labelme_by_level(
            IMG_PATH, H, W, roots, max_depth_calculated,
            min_region_area=min_region_area, n_dirs=n_dirs,
            min_frac=min_frac, max_frac=max_frac, fan_deg=fan_deg
        )
        logging.info(f"Export function reported creation of {len(created_json_paths)} JSON files.")
    except Exception as e:
        logging.error(f"Error exporting LabelMe JSONs: {e}")
        logging.error(traceback.format_exc())
        # Return empty list even if some files were created before error
        return IMG_PATH, H, W, []

    # ---- Final Visual Check (Optional based on save_plots) ----
    if save_plots and created_json_paths:
        try:
            logging.info("Generating final preview plot from exported JSON files...")
            fig_preview, ax_preview = plt.subplots(figsize=(10, 10))
            ax_preview.imshow(rgb)
            ax_preview.set_title("LabelMe Preview (All Levels - Vector Polygons)")
            num_depths = max_depth_calculated + 1
            # Use tab20 colormap
            cmap_func = plt.colormaps.get_cmap('tab20')
            colors = [cmap_func(i % cmap_func.N) for i in range(num_depths)] if num_depths > 0 else ['red']

            # Use the returned list of paths
            for json_file in created_json_paths:
                try:
                    level = int(Path(json_file).stem.split('_L')[-1])
                    lm_data = json.loads(Path(json_file).read_text())
                    level_idx = level % len(colors)
                    level_color_poly = colors[level_idx]
                    level_color_line = 'lime'

                    logging.debug(f"Plotting shapes from {json_file.name} (Level {level}) with color {level_color_poly}")
                    for sh in lm_data["shapes"]:
                        pts = np.asarray(sh["points"])
                        if sh["shape_type"] == "polygon" and pts.shape[0] >= 3:
                            ax_preview.fill(pts[:, 0], pts[:, 1], facecolor=list(level_color_poly[:3]) + [0.4], edgecolor=level_color_poly, linewidth=1.5)
                        elif sh["shape_type"] == "line" and pts.shape[0] == 2:
                            ax_preview.plot(pts[:, 0], pts[:, 1], color=level_color_line, linestyle='-', linewidth=1.5, alpha=0.8)
                except Exception as e:
                     logging.error(f"Failed to load or parse shapes from {json_file.name} for preview: {e}")

            ax_preview.axis('off')
            plt.tight_layout()
            preview_path = DATA_DIR / f"{IMG_PATH.stem}_labelme_preview_LEVELS.png"
            plt.savefig(preview_path)
            logging.info(f"Level-based LabelMe preview saved to {preview_path}")
            plt.close(fig_preview) # Close the figure
        except Exception as e:
            logging.error(f"Failed to generate LabelMe preview plot: {e}")

    plt.close('all') # Close any other potentially open figures
    logging.info("Segmentation and JSON export process finished.")
    return IMG_PATH, H, W, created_json_paths # Return paths

# Note: No `if __name__ == "__main__":` block needed here as it's meant to be imported.