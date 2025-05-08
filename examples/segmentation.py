#%%
# colour-PCA hierarchical merge (2025-04)
# Using original segmentation parameters, but new level-based JSON export.
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage import io, segmentation, color, filters, graph, measure, img_as_float
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─────────── Original Good Segmentation Parameters ───────────────────────
# Use the uploaded file path
IMG = "data/bird2.jpg" # <<< CHANGED TO USE UPLOADED FILE
DATA_DIR = Path(".") # Define a directory for output
DATA_DIR.mkdir(exist_ok=True) # Ensure output directory exists
IMG_PATH = Path(IMG) # Create Path object for image

# --- ADDED: Define and create a directory specifically for PNG previews ---
PREVIEW_DIR = DATA_DIR / "output_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True) # Create it if it doesn't exist

N_SEGMENTS = 1000 # SLIC initial segments (250 for sky2, 1000 for leaf2)
COMPACTNESS = 3 # SLIC compactness

ALPHA, BETA = 4.0, 12.0 # angle ‖ ΔLab cost weights (12 for sky2)
GAMMA = 6.0 # gradient penalty weight (3)
ETA = 0.5 # length penalty weight (ratio of PCA lengths)
L0 = 0.03 # threshold for "long" PCA axis in Lab space
THRESH = 6.0 # <<< Original merge threshold (2.8 for sky2,2.6 for leaf2)
# ───────────────────────────────────────────────────────────────────────

# ─────────── Polygon/JSON Generation Parameters (for Level-Based Export) ───
# --- Use LOW simplification here to avoid gaps between vector polygons ---
SIMPLIFY_PX = 0.1 # Polygon simplification tolerance
MIN_RING_AREA = 25 # px² – drop initial contour rings smaller than this
MIN_REGION_AREA = 50 # px² – drop final regions *after* punching out children

# --- Consider lowering MIN_REGION_AREA if background is still missing ---
# MIN_REGION_AREA = 10 # Example: Lower threshold

N_DIRS = 3 # how many direction chords per region
MIN_FRAC = .25 # min length of chord relative to max possible length
MAX_FRAC = .45 # max length of chord relative to max possible length
FAN_DEG = 75 # angular spread (± deg) around PCA axis for chords
# ───────────────────────────────────────────────────────────────────────


#%%
# ————————————————————————————————————————————————————————————————
# 1) Contours → valid exterior polygons (one per super-pixel label)
# (Using the robust version from later code)
# ————————————————————————————————————————————————————————————————
def polygons_from_labels(label_img, simplify_px: float = SIMPLIFY_PX, min_ring_area: float = MIN_RING_AREA):
    """
    Yields (region_id, shapely.Polygon) for each label in the image.
    Handles simplification and basic area filtering. Returns valid Polygon objects.
    """
    processed_ids = set()
    unique_labels = np.unique(label_img)
    logging.info(f"Processing {len(unique_labels)} unique labels found in image.")
    for rid in unique_labels:
        if rid < 0:
            logging.debug(f"Skipping region ID {rid} (negative).")
            continue # Skip background label if present (<0)

        # --- Added check: Should we process label 0? ---
        # Set process_label_zero = False if you are sure 0 is always unwanted background noise
        process_label_zero = True
        if rid == 0 and not process_label_zero:
             logging.info("Skipping region ID 0 based on configuration.")
             continue
        # --- End Added check ---

        if rid in processed_ids: continue # Should not happen with unique_labels, but safe check

        mask = label_img == rid
        # Check if mask contains any True values before finding contours
        if not np.any(mask):
            logging.debug(f"Skipping region ID {rid} (empty mask).")
            continue

        try:
            # Pad the mask slightly to help catch contours touching the boundary
            padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_mask.astype(float), 0.5)
            # Adjust contour coordinates back to original image space
            contours = [c - 1 for c in contours]

        except Exception as e:
            logging.error(f"find_contours failed for region {rid}: {e}")
            continue # Skip this region if contour finding fails

        if not contours:
            logging.debug(f"No contours found for region ID {rid}.")
            continue

        rings = []
        for cnt in contours:
            # Ensure contour has enough points for a polygon
            if len(cnt) < 4:
                 logging.debug(f"Skipping short contour ({len(cnt)} points) for region {rid}.")
                 continue
            # Convert coordinates (usually row, col) to (x, y) and create Polygon
            # Use buffer(0) to try and fix potential self-intersections or invalidities
            try:
                poly = Polygon(np.fliplr(cnt)).buffer(0)
            except Exception as e:
                logging.warning(f"Polygon creation/buffer failed for a contour in region {rid}: {e}")
                continue # Skip this contour

            if poly.is_empty or not poly.is_valid:
                 logging.debug(f"Skipping empty or invalid raw polygon for region {rid}.")
                 continue

            # Filter based on initial ring area
            if poly.area < min_ring_area:
                logging.debug(f"Skipping contour for region {rid} due to small area ({poly.area:.2f} < {min_ring_area}).")
                continue

            rings.append(poly)

        if not rings:
             logging.debug(f"No valid rings survived for region ID {rid} after initial filtering.")
             continue

        # Union potentially multiple contours/rings for the same region ID
        try:
            # Use buffer(0) again after union for robustness
            merged_geom = unary_union(rings).buffer(0)
        except Exception as e:
            logging.error(f"Unary union failed for region {rid}: {e}")
            continue # Skip this region if union fails

        # Handle the result of the union (Polygon or MultiPolygon)
        geoms_to_process = []
        if isinstance(merged_geom, Polygon):
            geoms_to_process.append(merged_geom)
        elif isinstance(merged_geom, MultiPolygon):
            # Process each polygon within the MultiPolygon
            geoms_to_process.extend(list(merged_geom.geoms))
        elif not merged_geom.is_empty:
            logging.warning(f"Unexpected geometry type {type(merged_geom)} after union for region {rid}. Skipping.")
            continue
        else: # Empty geometry after union
             logging.debug(f"Region {rid} resulted in empty geometry after union.")
             continue

        # Process each resulting geometry (usually one, but could be multiple if union resulted in MultiPolygon)
        processed_ids.add(rid) # Mark ID as processed
        for idx, geom in enumerate(geoms_to_process):
            if geom.is_empty or not geom.is_valid:
                logging.debug(f"Skipping empty/invalid geometry part {idx} for region {rid}.")
                continue

            try:
                # Simplify and buffer(0) again to ensure validity
                simplified_geom = geom.simplify(simplify_px, preserve_topology=True).buffer(0)
            except Exception as e:
                 logging.warning(f"Simplification/buffer failed for geometry part {idx} in region {rid}: {e}")
                 continue # Skip this part

            # Final check on area after simplification
            if not simplified_geom.is_empty and simplified_geom.is_valid and simplified_geom.area >= min_ring_area:
                logging.debug(f"Yielding polygon for region {rid} (part {idx}) with area {simplified_geom.area:.2f}.")
                # Yield the original region ID and the final polygon
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
    """Builds the containment hierarchy (forest) from polygons.
    Includes a small buffer in contains check for robustness.
    Returns roots and max_depth.
    """
    nodes = {rid: Region(rid, g) for rid, g in rid2poly.items()}
    logging.info(f"Building forest for {len(nodes)} regions.")
    if not nodes:
        return [], -1

    # --- Add a small buffer for contains check ---
    CONTAINS_BUFFER = 1e-6 # A very small positive buffer

    # O(n²) containment check
    for r1, r2 in combinations(nodes.values(), 2):
        # Ensure geometries are valid before buffering/checking
        if not r1.orig_geom.is_valid: r1.orig_geom = r1.orig_geom.buffer(0)
        if not r2.orig_geom.is_valid: r2.orig_geom = r2.orig_geom.buffer(0)
        # Skip if still invalid after attempting to fix
        if not r1.orig_geom.is_valid or not r2.orig_geom.is_valid:
            logging.warning(f"Skipping invalid geometry during contains check for region {r1.id} or {r2.id}")
            continue

        try:
            # Use buffer on potential parent
            # Ensure the potential child is strictly smaller for contains check
            contains_r2 = r1.orig_geom.area > r2.orig_geom.area and r1.orig_geom.buffer(CONTAINS_BUFFER).contains(r2.orig_geom)
            contains_r1 = r2.orig_geom.area > r1.orig_geom.area and r2.orig_geom.buffer(CONTAINS_BUFFER).contains(r1.orig_geom)

        except Exception as e:
            logging.error(f"Contains check failed between region {r1.id} and {r2.id}: {e}", exc_info=False) # Reduce log noise
            continue

        # Basic check if one contains the other
        if contains_r2: # r1 might be parent of r2
             # Check if r2 already has a parent, if so, is r1 closer (contains current parent)?
             if r2.parent is None or r2.parent.orig_geom.contains(r1.orig_geom):
                 # Avoid direct cycles (though area check should prevent most)
                 if r1.parent is None or r1.parent.id != r2.id:
                      # Remove r2 from old parent's children if necessary
                      if r2.parent is not None:
                          try: r2.parent.children.remove(r2)
                          except ValueError: pass # Already removed or not there
                      r1.children.append(r2)
                      r2.parent = r1

        elif contains_r1: # r2 might be parent of r1
            # Check if r1 already has a parent, if so, is r2 closer?
             if r1.parent is None or r1.parent.orig_geom.contains(r2.orig_geom):
                 if r2.parent is None or r2.parent.id != r1.id:
                      if r1.parent is not None:
                          try: r1.parent.children.remove(r1)
                          except ValueError: pass
                      r2.children.append(r1)
                      r1.parent = r2

    # Find roots and calculate depth
    roots = [n for n in nodes.values() if n.parent is None]
    logging.info(f"Found {len(roots)} root regions initially.")

    # Calculate depth using BFS from roots
    processed_depths = set()
    queue = [(root, 0) for root in roots]
    max_calculated_depth = -1
    visited_in_queue = {r.id for r in roots} # Track nodes added to queue

    while queue:
        node, d = queue.pop(0)
        # Skip if already processed (e.g., complex graph structure, though should be tree/forest)
        if node.id in processed_depths: continue

        node.depth = d
        max_calculated_depth = max(max_calculated_depth, d)
        processed_depths.add(node.id)

        for ch in node.children:
            # Add child to queue only if not already visited/added
            if ch.id not in visited_in_queue:
                 queue.append((ch, d + 1))
                 visited_in_queue.add(ch.id)

    if len(processed_depths) != len(nodes):
        logging.warning(f"Depth calculation issue: Processed {len(processed_depths)} depths, but expected {len(nodes)} nodes.")
        # Find nodes without depth
        nodes_without_depth = [nid for nid in nodes if nodes[nid].depth == -1]
        logging.warning(f"Nodes without assigned depth: {nodes_without_depth}")
        # Assign depth 0 to any remaining orphans to ensure export
        for nid in nodes_without_depth:
             if nodes[nid].depth == -1:
                  nodes[nid].depth = 0
                  if nodes[nid].parent is None and nodes[nid] not in roots:
                       roots.append(nodes[nid])
                  logging.warning(f"Assigned depth 0 to orphan node {nid}.")
        max_calculated_depth = max(max_calculated_depth, 0) # Ensure max depth is at least 0


    logging.info(f"Final root count: {len(roots)}. Max calculated depth: {max_calculated_depth}")
    return roots, max_calculated_depth

def inner_chords(poly: Polygon, forbidden_polys: list[Polygon], rng: np.random.Generator, n_dirs=N_DIRS, min_frac=MIN_FRAC, max_frac=MAX_FRAC, fan_deg=FAN_DEG):
    if poly.is_empty or not poly.is_valid or poly.area < 1e-6: return []
    # Ensure polygon is valid before proceeding
    current_poly = poly if poly.is_valid else poly.buffer(0)
    if not current_poly.is_valid or current_poly.is_empty:
        logging.warning(f"Skipping inner_chords for invalid/empty polygon (Area: {poly.area:.2f})")
        return []

    # Get representative point safely
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

    # Calculate PCA axis
    try:
        xy = np.asarray(current_poly.exterior.coords) - (Cx, Cy)
        if xy.shape[0] <= 2: return [] # Not enough points for SVD
        # Use only exterior points, ignore holes for axis calculation
        _, _, Vt = np.linalg.svd(xy[:-1], full_matrices=False)
        axis = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-9) # Normalized first principal component
    except Exception as e: # SVD might fail for degenerate cases
        logging.warning(f"SVD failed for PCA axis: {e}. Using random axis.")
        axis = rng.random(2) * 2 - 1 # Random vector
        axis = axis / (np.linalg.norm(axis) + 1e-9) # Normalize

    # --- ADDED: Identifier for logging this specific polygon call ---
    poly_id_str = f"poly_area_{poly.area:.1f}" # Use area as an approximate ID

    segs, tries = [], 0
    max_tries = n_dirs * 200 # Increase max tries slightly

    while len(segs) < n_dirs and tries < max_tries:
        tries += 1
        # Generate random direction within fan around PCA axis
        angle_rad = np.deg2rad(rng.uniform(-fan_deg, fan_deg))
        rot_matrix = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])
        v = rot_matrix @ axis # Rotated direction vector

        # Create a long line through the representative point in the chosen direction
        bounds = current_poly.bounds # (minx, miny, maxx, maxy)
        if not (len(bounds)==4 and all(isinstance(b,float) for b in bounds)): continue # Invalid bounds
        diag_len = np.sqrt((bounds[2]-bounds[0])**2 + (bounds[3]-bounds[1])**2)
        far = diag_len*1.5 + 10 # Extend line well beyond polygon bounds
        line = LineString([(Cx - far * v[0], Cy - far * v[1]), (Cx + far * v[0], Cy + far * v[1])])

        # Intersect the line with the polygon to get the chord
        try:
            chord = line.intersection(current_poly)
        except Exception as e:
             # --- MODIFIED: Added logging ---
             logging.debug(f"[{poly_id_str} Try {tries}] Line intersection failed: {e}")
             continue # Skip if intersection calculation fails

        # Process only if intersection is a valid LineString
        # --- MODIFIED: Added logging for specific rejection reasons (can be verbose) ---
        if chord.is_empty:
            # logging.debug(f"[{poly_id_str} Try {tries}] Chord is empty.")
            continue
        if not isinstance(chord, LineString) or len(chord.coords)<2:
             logging.debug(f"[{poly_id_str} Try {tries}] Chord not valid LineString (Type: {type(chord)}).")
             continue

        # Calculate chord length and sub-segment parameters
        c0, c1 = map(np.asarray, chord.coords)
        full_len = np.linalg.norm(c1 - c0)
        if full_len < 1e-3:
            # logging.debug(f"[{poly_id_str} Try {tries}] Chord too short ({full_len:.3f}).")
            continue # Chord too short

        min_len = min_frac * full_len
        max_len = max_frac * full_len
        if max_len <= min_len or min_len < 0: continue # Invalid length range

        # Calculate the segment within the chord
        target_len = rng.uniform(min_len, max_len)
        start_offset = rng.uniform(0, full_len - target_len) if full_len > target_len else 0
        end_offset = start_offset + target_len
        unit_vec = (c1 - c0) / (full_len + 1e-9)

        # Calculate segment points with a small inward padding
        pad = min(0.5, target_len * 0.01, full_len * 0.005) # Reduced padding slightly
        p0 = c0 + unit_vec * (start_offset + pad)
        p1 = c0 + unit_vec * (end_offset - pad)

        # Ensure segment still has positive length after padding
        if np.linalg.norm(p1 - p0) < 1e-3:
            # logging.debug(f"[{poly_id_str} Try {tries}] Segment too short after padding.")
            continue

        seg = LineString([tuple(p0), tuple(p1)])

        # Final checks: segment must be within the polygon and not intersect forbidden areas
        try:
            # Check containment with a small negative buffer for robustness
            is_contained = current_poly.buffer(-1e-6).contains(seg)
            if not is_contained:
                 # --- MODIFIED: Added logging ---
                 logging.debug(f"[{poly_id_str} Try {tries}] Segment not contained in parent polygon.")
                 continue # Skip if not contained

            intersects_forbidden = False
            intersecting_forbidden_id = None # Track which forbidden poly caused rejection
            for i, f_poly in enumerate(forbidden_polys):
                # Ensure forbidden polygons are valid before checking intersection
                valid_f_poly = f_poly if f_poly.is_valid else f_poly.buffer(0)
                if not valid_f_poly.is_empty and seg.intersects(valid_f_poly):
                    intersects_forbidden = True
                    # Try to get an idea of which forbidden polygon it was
                    intersecting_forbidden_id = f"index {i}, area {valid_f_poly.area:.1f}"
                    break # No need to check other forbidden polys

            if intersects_forbidden:
                 # --- MODIFIED: Added logging ---
                 logging.debug(f"[{poly_id_str} Try {tries}] Segment intersects forbidden polygon ({intersecting_forbidden_id}).")
                 continue # Skip if intersects forbidden

            # If we reach here, the segment is valid
            segs.append([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]])

        except Exception as e:
             # --- MODIFIED: Added logging ---
             logging.debug(f"[{poly_id_str} Try {tries}] Final segment check failed: {e}")
             continue # Skip if final checks fail
    # --- End MODIFIED ---

    # --- ADDED: Log if not enough segments were found ---
    if len(segs) < n_dirs:
        logging.warning(f"[{poly_id_str}] Only generated {len(segs)}/{n_dirs} lines after {tries} tries.")

    return segs
# <<< Replace the existing export_labelme_by_level function in your segmentation script with this >>>

def export_labelme_by_level(img_path_obj: Path, H: int, W: int, roots: list[Region], calculated_max_depth: int):
    img_path_str = str(img_path_obj); img_basename = img_path_obj.name; out_dir = img_path_obj.parent; stem = img_path_obj.stem
    all_nodes = [node for r in roots for node in r]
    if not all_nodes: logging.warning("No regions found for JSON export."); return calculated_max_depth

    level2nodes = defaultdict(list); logging.info("Punching out children and filtering by MIN_REGION_AREA..."); nodes_removed_count = 0
    final_nodes_for_export = []

    # --- Step 1: Calculate Final Geometries (Punch-out) and Filter ---
    for n in all_nodes:
        punched_geom = n.orig_geom
        if not punched_geom or not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)
        if not punched_geom or not punched_geom.is_valid or punched_geom.is_empty:
             logging.warning(f"Region {n.id} has invalid/empty original geometry. Skipping.")
             nodes_removed_count += 1; n.final_geom = None; continue

        if n.children:
            valid_child_geoms = []
            for c in n.children:
                child_geom = c.orig_geom
                if not child_geom or not child_geom.is_valid: child_geom = child_geom.buffer(0)
                if child_geom and child_geom.is_valid and not child_geom.is_empty:
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

        if not punched_geom or punched_geom.is_empty or not punched_geom.is_valid or punched_geom.area < MIN_REGION_AREA:
            logging.info(f"Region {n.id} removed after punching/filtering (Area: {punched_geom.area if punched_geom else 0:.2f} < {MIN_REGION_AREA} or invalid/empty).")
            nodes_removed_count += 1; n.final_geom = None; continue
        else:
             n.final_geom = punched_geom
             if n.depth >= 0: final_nodes_for_export.append(n)

    if nodes_removed_count > 0: logging.info(f"Total {nodes_removed_count} regions removed before export.")
    if not final_nodes_for_export: logging.warning("No regions survived filtering for JSON export."); return calculated_max_depth

    for n in final_nodes_for_export: level2nodes[n.depth].append(n)

    # --- Step 2: Exporting ---
    img_b64 = None; num_exported_levels = 0
    logging.info(f"Exporting JSON levels 0 to {calculated_max_depth} for surviving regions.")

    for lvl in range(calculated_max_depth + 1):
        nodes_at_level = level2nodes.get(lvl, [])
        if not nodes_at_level: continue

        shapes = []; num_polys_at_level = 0
        # Get final geometries of siblings at the current level
        sibling_final_geoms = {sib.id: sib.final_geom for sib in nodes_at_level if sib.final_geom}

        for n in nodes_at_level:
            if n.final_geom is None: continue # Should have been filtered, but check again

            num_polys_at_level += 1
            current_final_geom = n.final_geom
            geoms_to_add_for_shape = [] # Polygon parts for the JSON shape field

            if isinstance(current_final_geom, Polygon):
                geoms_to_add_for_shape.append(current_final_geom)
            elif isinstance(current_final_geom, MultiPolygon):
                geoms_to_add_for_shape.extend(p for p in current_final_geom.geoms if p.area >= MIN_REGION_AREA and p.is_valid)

            # --- Chord Generation Logic (Conditional based on Level) ---
            chord_segments_for_node = []
            rng = np.random.default_rng(seed=n.id + lvl)

            if lvl == 0:
                # For Level 0 (typically background), generate chords on the FINAL geometry.
                # Forbidden list only contains siblings' FINAL geometries.
                logging.debug(f"Region {n.id} L{lvl}: Generating chords on FINAL geometry.")
                forbidden_lvl0 = [g for k, g in sibling_final_geoms.items() if k != n.id and g]
                if n.final_geom and n.final_geom.is_valid and not n.final_geom.is_empty:
                    # Need to handle if final_geom is MultiPolygon for chord generation
                    geom_for_chords = n.final_geom
                    if isinstance(geom_for_chords, MultiPolygon):
                        # Simplification: Use the largest part of the MultiPolygon for chord generation
                        valid_parts = [p for p in geom_for_chords.geoms if p.is_valid and p.area > 1e-3]
                        if valid_parts:
                            geom_for_chords = max(valid_parts, key=lambda p: p.area)
                        else:
                            geom_for_chords = None # Cannot generate chords
                    
                    if geom_for_chords and isinstance(geom_for_chords, Polygon):
                         chord_segments_for_node = inner_chords(geom_for_chords, forbidden_lvl0, rng)
                    else:
                         logging.warning(f"Region {n.id} L{lvl}: Could not determine valid Polygon from final_geom to generate chords.")

                else:
                     logging.warning(f"Region {n.id} L{lvl}: Skipping chord generation due to invalid/empty final geometry.")

            else: # lvl > 0
                # For inner levels, generate chords on ORIGINAL geometry and validate against FINAL.
                logging.debug(f"Region {n.id} L{lvl}: Generating chords on ORIGINAL geometry, validating against FINAL.")
                if n.orig_geom and n.orig_geom.is_valid and not n.orig_geom.is_empty:
                    # Forbidden list includes siblings' FINAL geoms and children's ORIGINAL geoms
                    forbidden_inner = [g for k, g in sibling_final_geoms.items() if k != n.id and g] \
                                    + [d.orig_geom for d in n if d.id != n.id and d.orig_geom and d.orig_geom.is_valid and not d.orig_geom.is_empty]

                    candidate_segments = inner_chords(n.orig_geom, forbidden_inner, rng)

                    # Validate against FINAL geometry
                    for seg_coords in candidate_segments:
                        try:
                            line_seg = LineString(seg_coords)
                            # Use a small positive buffer for contains check robustness
                            if n.final_geom.buffer(1e-6).contains(line_seg):
                                chord_segments_for_node.append(seg_coords)
                            else:
                                logging.debug(f"Region {n.id} L{lvl}: Chord segment rejected (not contained in final geom).")
                        except Exception as e:
                            logging.debug(f"Region {n.id} L{lvl}: Error checking chord segment containment: {e}")
                else:
                    logging.warning(f"Region {n.id} L{lvl}: Skipping chord generation due to invalid/missing original geometry.")
            # --- End Chord Generation Logic ---

            # Add polygon shapes and validated chord shapes
            for poly_part in geoms_to_add_for_shape:
                if poly_part.is_empty or not poly_part.is_valid or poly_part.exterior is None or len(poly_part.exterior.coords) < 4:
                     logging.warning(f"Skipping invalid/small polygon part for region {n.id} at level {lvl}.")
                     continue
                pts = np.asarray(poly_part.exterior.coords[:-1]).tolist()
                shapes.append({"label":f"region-{n.id}", "points":pts, "group_id":None, "shape_type":"polygon", "flags":{}})

            for seg_coords in chord_segments_for_node:
                 shapes.append({"label":f"direction-{n.id}", "points":seg_coords, "group_id":None, "shape_type":"line", "flags":{}})

        # --- Write JSON File for the Level ---
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
            poly_count_this_level = sum(1 for s in shapes if s['shape_type'] == 'polygon')
            logging.info(f"✓ Wrote {out_path.name} ({poly_count_this_level} polygons, {len(shapes)} total shapes)")
            num_exported_levels += 1
        except Exception as e: logging.error(f"Failed to write JSON file {out_path}: {e}")

    if num_exported_levels == 0: logging.warning("No JSON files were exported for any level.")
    else: logging.info(f"Exported {num_exported_levels} JSON level file(s).")

    return calculated_max_depth
"""
def export_labelme_by_level(img_path_obj: Path, H: int, W: int, roots: list[Region], calculated_max_depth: int):
    img_path_str = str(img_path_obj); img_basename = img_path_obj.name; out_dir = img_path_obj.parent; stem = img_path_obj.stem
    all_nodes = [node for r in roots for node in r]
    if not all_nodes: logging.warning("No regions found for JSON export."); return calculated_max_depth

    level2nodes = defaultdict(list); logging.info("Punching out children and filtering by MIN_REGION_AREA..."); nodes_removed_count = 0
    final_nodes_for_export = []

    # Iterate through all nodes to punch out children and apply MIN_REGION_AREA filter
    # (Punch-out logic remains the same - determines n.final_geom)
    for n in all_nodes:
        punched_geom = n.orig_geom # Start with the original polygon for this region
        if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)
        if not punched_geom.is_valid or punched_geom.is_empty:
             logging.warning(f"Region {n.id} has invalid/empty original geometry. Skipping.")
             nodes_removed_count += 1
             n.final_geom = None # Mark as removed
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
                    punched_geom = n.orig_geom
                    if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)
        if punched_geom.is_empty or not punched_geom.is_valid or punched_geom.area < MIN_REGION_AREA:
            logging.info(f"Region {n.id} removed after punching/filtering (Area: {punched_geom.area:.2f} < {MIN_REGION_AREA} or invalid/empty).")
            nodes_removed_count += 1
            n.final_geom = None # Mark as removed
            continue
        else:
             n.final_geom = punched_geom
             if n.depth >= 0:
                final_nodes_for_export.append(n)

    if nodes_removed_count > 0: logging.info(f"Total {nodes_removed_count} regions removed before export due to area/validity constraints.")
    if not final_nodes_for_export: logging.warning("No regions survived filtering for JSON export."); return calculated_max_depth

    for n in final_nodes_for_export:
         level2nodes[n.depth].append(n)

    # --- Exporting ---
    img_b64 = None
    num_exported_levels = 0
    logging.info(f"Exporting JSON levels 0 to {calculated_max_depth} for surviving regions.")

    for lvl in range(calculated_max_depth + 1):
        nodes_at_level = level2nodes.get(lvl, [])
        if not nodes_at_level: continue

        shapes = []
        sibling_geoms = {n.id: n.final_geom for n in nodes_at_level if n.final_geom}
        num_polys_at_level = 0

        for n in nodes_at_level:
            if n.final_geom is None or n.final_geom.is_empty or not n.final_geom.is_valid: continue

            num_polys_at_level += 1
            current_final_geom = n.final_geom # Use the final, punched-out geometry for export shape
            geoms_to_add = []

            if isinstance(current_final_geom, Polygon):
                geoms_to_add.append(current_final_geom)
            elif isinstance(current_final_geom, MultiPolygon):
                geoms_to_add.extend(p for p in current_final_geom.geoms if p.area >= MIN_REGION_AREA and p.is_valid)

            # --- Chord Generation Logic Modified ---
            chord_segments_for_node = []
            # Try generating chords based on the original geometry first
            if n.orig_geom and n.orig_geom.is_valid and not n.orig_geom.is_empty:
                rng = np.random.default_rng(seed=n.id + lvl)
                forbidden = [g for k, g in sibling_geoms.items() if k != n.id and g] \
                          + [d.orig_geom for d in n if d.id != n.id and d.orig_geom.is_valid and not d.orig_geom.is_empty]

                # Generate chords using the ORIGINAL geometry
                candidate_segments = inner_chords(n.orig_geom, forbidden, rng)

                # Check if generated segments lie within the FINAL (punched-out) geometry
                for seg_coords in candidate_segments:
                    try:
                        line_seg = LineString(seg_coords)
                        # Check containment against the overall final geometry of the node
                        if n.final_geom.buffer(1e-6).contains(line_seg):
                            chord_segments_for_node.append(seg_coords)
                        else:
                            logging.debug(f"Region {n.id} L{lvl}: Chord segment rejected (not contained in final geom).")
                    except Exception as e:
                        logging.debug(f"Region {n.id} L{lvl}: Error checking chord segment containment: {e}")
            else:
                logging.warning(f"Region {n.id} L{lvl}: Skipping chord generation due to invalid/missing original geometry.")
            # --- End Chord Generation Logic Modified ---


            # Add polygon shapes and validated chord shapes
            for poly_part in geoms_to_add:
                if poly_part.is_empty or not poly_part.is_valid or poly_part.exterior is None or len(poly_part.exterior.coords) < 4:
                     logging.warning(f"Skipping invalid/small polygon part for region {n.id} at level {lvl}.")
                     continue

                # Add Polygon Shape
                pts = np.asarray(poly_part.exterior.coords[:-1]).tolist()
                shapes.append({"label":f"region-{n.id}", "points":pts, "group_id":None, "shape_type":"polygon", "flags":{}})

            # Add the validated chord segments for this node
            for seg_coords in chord_segments_for_node:
                 shapes.append({"label":f"direction-{n.id}", "points":seg_coords, "group_id":None, "shape_type":"line", "flags":{}})
                 # Limit number of lines added per node if desired (e.g. if inner_chords generated many)
                 # if len([s for s in shapes if s['label']==f"direction-{n.id}"]) >= N_DIRS: break


        # Write JSON file if shapes were generated for this level
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
            # Count polygons by checking labels, as one node might result in MultiPolygon parts
            poly_count_this_level = sum(1 for s in shapes if s['shape_type'] == 'polygon')
            logging.info(f"✓ Wrote {out_path.name} ({poly_count_this_level} polygons, {len(shapes)} total shapes)")
            num_exported_levels += 1
        except Exception as e: logging.error(f"Failed to write JSON file {out_path}: {e}")

    if num_exported_levels == 0: logging.warning("No JSON files were exported for any level.")
    else: logging.info(f"Exported {num_exported_levels} JSON level file(s).")

    return calculated_max_depth
"""
"""
def inner_chords(poly: Polygon, forbidden_polys: list[Polygon], rng: np.random.Generator, n_dirs=N_DIRS, min_frac=MIN_FRAC, max_frac=MAX_FRAC, fan_deg=FAN_DEG):
    if poly.is_empty or not poly.is_valid or poly.area < 1e-6: return []
    # Ensure polygon is valid before proceeding
    current_poly = poly if poly.is_valid else poly.buffer(0)
    if not current_poly.is_valid or current_poly.is_empty:
        logging.warning(f"Skipping inner_chords for invalid/empty polygon (Area: {poly.area:.2f})")
        return []

    # Get representative point safely
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


    # Calculate PCA axis
    try:
        xy = np.asarray(current_poly.exterior.coords) - (Cx, Cy)
        if xy.shape[0] <= 2: return [] # Not enough points for SVD
        # Use only exterior points, ignore holes for axis calculation
        _, _, Vt = np.linalg.svd(xy[:-1], full_matrices=False)
        axis = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-9) # Normalized first principal component
    except Exception as e: # SVD might fail for degenerate cases
        logging.warning(f"SVD failed for PCA axis: {e}. Using random axis.")
        axis = rng.random(2) * 2 - 1 # Random vector
        axis = axis / (np.linalg.norm(axis) + 1e-9) # Normalize


    segs, tries = [], 0
    max_tries = n_dirs * 20 # Increase max tries slightly

    while len(segs) < n_dirs and tries < max_tries:
        tries += 1
        # Generate random direction within fan around PCA axis
        angle_rad = np.deg2rad(rng.uniform(-fan_deg, fan_deg))
        rot_matrix = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])
        v = rot_matrix @ axis # Rotated direction vector

        # Create a long line through the representative point in the chosen direction
        bounds = current_poly.bounds # (minx, miny, maxx, maxy)
        if not (len(bounds)==4 and all(isinstance(b,float) for b in bounds)): continue # Invalid bounds
        diag_len = np.sqrt((bounds[2]-bounds[0])**2 + (bounds[3]-bounds[1])**2)
        far = diag_len*1.5 + 10 # Extend line well beyond polygon bounds
        line = LineString([(Cx - far * v[0], Cy - far * v[1]), (Cx + far * v[0], Cy + far * v[1])])

        # Intersect the line with the polygon to get the chord
        try:
            chord = line.intersection(current_poly)
        except Exception as e:
             logging.debug(f"Line intersection failed: {e}")
             continue # Skip if intersection calculation fails

        # Process only if intersection is a valid LineString
        if chord.is_empty or not isinstance(chord, LineString) or len(chord.coords)<2: continue

        # Calculate chord length and sub-segment parameters
        c0, c1 = map(np.asarray, chord.coords)
        full_len = np.linalg.norm(c1 - c0)
        if full_len < 1e-3: continue # Chord too short

        min_len = min_frac * full_len
        max_len = max_frac * full_len
        if max_len <= min_len or min_len < 0: continue # Invalid length range

        # Calculate the segment within the chord
        target_len = rng.uniform(min_len, max_len)
        start_offset = rng.uniform(0, full_len - target_len) if full_len > target_len else 0
        end_offset = start_offset + target_len
        unit_vec = (c1 - c0) / (full_len + 1e-9)

        # Calculate segment points with a small inward padding
        pad = min(0.5, target_len * 0.01, full_len * 0.005) # Reduced padding slightly
        p0 = c0 + unit_vec * (start_offset + pad)
        p1 = c0 + unit_vec * (end_offset - pad)

        # Ensure segment still has positive length after padding
        if np.linalg.norm(p1 - p0) < 1e-3: continue

        seg = LineString([tuple(p0), tuple(p1)])

        # Final checks: segment must be within the polygon and not intersect forbidden areas
        try:
            # Check containment with a small negative buffer for robustness
            is_contained = current_poly.buffer(-1e-6).contains(seg)
            intersects_forbidden = False
            if is_contained:
                for f_poly in forbidden_polys:
                    # Ensure forbidden polygons are valid before checking intersection
                    valid_f_poly = f_poly if f_poly.is_valid else f_poly.buffer(0)
                    if not valid_f_poly.is_empty and seg.intersects(valid_f_poly):
                        intersects_forbidden = True
                        break # No need to check other forbidden polys

            if is_contained and not intersects_forbidden:
                # Append segment coordinates if valid
                segs.append([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]])

        except Exception as e:
             logging.debug(f"Final segment check failed: {e}")
             continue # Skip if final checks fail

    # if len(segs) < n_dirs: logging.warning(f"Only {len(segs)}/{n_dirs} lines for polygon area {poly.area:.2f}")
    return segs

def export_labelme_by_level(img_path_obj: Path, H: int, W: int, roots: list[Region], calculated_max_depth: int):
    img_path_str = str(img_path_obj); img_basename = img_path_obj.name; out_dir = img_path_obj.parent; stem = img_path_obj.stem
    all_nodes = [node for r in roots for node in r]
    if not all_nodes: logging.warning("No regions found for JSON export."); return calculated_max_depth

    level2nodes = defaultdict(list); logging.info("Punching out children and filtering by MIN_REGION_AREA..."); nodes_removed_count = 0
    final_nodes_for_export = []

    # Iterate through all nodes to punch out children and apply MIN_REGION_AREA filter
    for n in all_nodes:
        punched_geom = n.orig_geom # Start with the original polygon for this region

        # Ensure the original geometry is valid before proceeding
        if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)
        if not punched_geom.is_valid or punched_geom.is_empty:
             logging.warning(f"Region {n.id} has invalid/empty original geometry. Skipping.")
             nodes_removed_count += 1
             n.final_geom = None # Mark as removed
             continue

        # Subtract valid child geometries
        if n.children:
            valid_child_geoms = []
            for c in n.children:
                child_geom = c.orig_geom
                if not child_geom.is_valid: child_geom = child_geom.buffer(0)
                if child_geom.is_valid and not child_geom.is_empty:
                    valid_child_geoms.append(child_geom)

            if valid_child_geoms:
                try:
                    # Union all valid children first
                    kids_union = unary_union(valid_child_geoms)
                    if not kids_union.is_valid: kids_union = kids_union.buffer(0)

                    # Punch out the unioned children if valid
                    if kids_union.is_valid and not kids_union.is_empty:
                        punched_geom = punched_geom.difference(kids_union)
                        # Apply buffer(0) after difference to clean up potential issues
                        if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)

                except Exception as e:
                    logging.error(f"Punch out failed for region {n.id}: {e}. Using original geometry.")
                    # Fallback to original geometry if punchout fails
                    punched_geom = n.orig_geom
                    if not punched_geom.is_valid: punched_geom = punched_geom.buffer(0)


        # Final check on the resulting geometry (after punchout)
        if punched_geom.is_empty or not punched_geom.is_valid or punched_geom.area < MIN_REGION_AREA:
            logging.info(f"Region {n.id} removed after punching/filtering (Area: {punched_geom.area:.2f} < {MIN_REGION_AREA} or invalid/empty).")
            nodes_removed_count += 1
            n.final_geom = None # Mark as removed
            continue
        else:
             # Store the final geometry and add node to list if it survived
             n.final_geom = punched_geom
             if n.depth >= 0: # Only consider nodes with valid depth
                final_nodes_for_export.append(n)


    if nodes_removed_count > 0: logging.info(f"Total {nodes_removed_count} regions removed before export due to area/validity constraints.")
    if not final_nodes_for_export: logging.warning("No regions survived filtering for JSON export."); return calculated_max_depth


    # Group surviving nodes by their calculated depth
    for n in final_nodes_for_export:
         level2nodes[n.depth].append(n)

    # --- Exporting ---
    img_b64 = None # Image data cache
    num_exported_levels = 0
    logging.info(f"Exporting JSON levels 0 to {calculated_max_depth} for surviving regions.")

    for lvl in range(calculated_max_depth + 1):
        nodes_at_level = level2nodes.get(lvl, [])
        if not nodes_at_level:
            logging.debug(f"No regions to export at level {lvl}.")
            continue # Skip levels with no surviving nodes

        shapes = [] # Shapes for the current level's JSON file
        # Create a dictionary of geometries at the current level for chord generation context
        sibling_geoms = {n.id: n.final_geom for n in nodes_at_level if n.final_geom}

        num_polys_at_level = 0
        for n in nodes_at_level:
            # Double-check if final_geom exists (should always exist here)
            if n.final_geom is None or n.final_geom.is_empty or not n.final_geom.is_valid:
                 continue

            num_polys_at_level += 1
            current_poly = n.final_geom # Use the final, punched-out geometry
            geoms_to_add = [] # Handles MultiPolygons resulting from punch-out

            if isinstance(current_poly, Polygon):
                geoms_to_add.append(current_poly)
            elif isinstance(current_poly, MultiPolygon):
                # Add only parts of the MultiPolygon that meet the area requirement
                geoms_to_add.extend(p for p in current_poly.geoms if p.area >= MIN_REGION_AREA and p.is_valid)

            # Generate shapes for each valid part of the region's geometry
            for poly_part in geoms_to_add:
                # Basic check for polygon part validity
                if poly_part.is_empty or not poly_part.is_valid or poly_part.exterior is None or len(poly_part.exterior.coords) < 4:
                     logging.warning(f"Skipping invalid/small polygon part for region {n.id} at level {lvl}.")
                     continue

                # Extract exterior points for LabelMe format (remove duplicate end point)
                pts = np.asarray(poly_part.exterior.coords[:-1]).tolist()
                shapes.append({"label":f"region-{n.id}", "points":pts, "group_id":None, "shape_type":"polygon", "flags":{}})

                # --- Generate Inner Chords ---
                rng = np.random.default_rng(seed=n.id + lvl) # Seed RNG for reproducibility

                # Define forbidden regions for chords: siblings at the same level + direct descendants' original shapes
                forbidden = [g for k, g in sibling_geoms.items() if k != n.id and g] \
                          + [d.orig_geom for d in n if d.id != n.id and d.orig_geom.is_valid and not d.orig_geom.is_empty]

                for seg in inner_chords(poly_part, forbidden, rng):
                    shapes.append({"label":f"direction-{n.id}", "points":seg, "group_id":None, "shape_type":"line", "flags":{}})


        # Write JSON file if shapes were generated for this level
        if not shapes:
             logging.info(f"No shapes generated for level {lvl}, skipping JSON file.")
             continue

        # Load image data only once if needed
        if img_b64 is None:
            logging.info("Reading image data for embedding...")
            try:
                img_b64 = base64.b64encode(img_path_obj.read_bytes()).decode("utf-8")
            except Exception as e:
                logging.error(f"Failed to read/encode image file {img_path_obj}: {e}")
                img_b64 = "" # Set empty if reading fails

        # Prepare LabelMe data structure
        lm_data = {"version":"5.2.1", "flags":{}, "shapes":shapes, "imagePath":img_basename, "imageData":img_b64, "imageHeight":H, "imageWidth":W}

        # Define output path and write JSON
        out_path = out_dir / f"{stem}_labelme_L{lvl}.json"
        try:
            out_path.write_text(json.dumps(lm_data, indent=2))
            logging.info(f"✓ Wrote {out_path.name} ({num_polys_at_level} polygons, {len(shapes)} total shapes)")
            num_exported_levels += 1
        except Exception as e:
            logging.error(f"Failed to write JSON file {out_path}: {e}")

    if num_exported_levels == 0:
        logging.warning("No JSON files were exported for any level.")
    else:
        logging.info(f"Exported {num_exported_levels} JSON level file(s).")

    return calculated_max_depth # Return max depth for consistency
# --- End Placeholder ---
"""

#%%
# —————————————————— Segmentation Helpers & Execution ————————————————————
# (Using original helpers & execution logic to get lbl2)

def pca_axis(lab_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the mean and endpoints along the first principal component axis."""
    if lab_pixels.shape[0] < 2:
        mu = lab_pixels.mean(0) if lab_pixels.shape[0] > 0 else np.zeros(3)
        return mu, mu # Return mean twice if not enough points

    μ = lab_pixels.mean(0)
    try:
        # Center data, perform SVD
        _, S, Vt = np.linalg.svd(lab_pixels - μ, full_matrices=False)
        v1 = Vt[0] # First principal component vector
        # Standard deviation along the first principal component
        σ1 = S[0] / np.sqrt(max(1, lab_pixels.shape[0] - 1))
        # Return endpoints: mean ± std_dev * principal_vector
        return μ - v1 * σ1, μ + v1 * σ1
    except np.linalg.LinAlgError:
        logging.warning("SVD failed in pca_axis, returning mean twice.")
        return μ, μ # Fallback to mean if SVD fails

def angle_penalty(v1_norm, v2_norm, ℓ1, ℓ2):
    """Calculates penalty based on angle between PCA axes and their lengths."""
    if ℓ1 < 1e-6 or ℓ2 < 1e-6: return 1.0 # No penalty if one vector is zero length
    # Cosine similarity -> angle
    cos_theta = abs(v1_norm @ v2_norm) # Use abs for angle between 0 and 90 deg
    theta = np.arccos(np.clip(cos_theta, 0, 1))
    # Lengths
    ℓm = max(ℓ1, ℓ2)
    ℓmi = min(ℓ1, ℓ2)
    # Penalty depends on whether axes are "long" (above L0)
    if ℓm < L0: return np.exp(-ALPHA * theta * theta) # Both short
    elif ℓmi < L0: return np.exp(-0.5 * ALPHA * theta * theta) # One short, one long
    return 1.0 # Both long, angle penalty might be less important (currently no penalty)


# Global dictionaries (matching original structure)
LMEAN = {}
PCA = {}
GRAD = {}
lab_sum = {}
lab_npix = {}

def affinity(u, v):
    """Calculates the affinity (similarity) between two regions u and v."""
    # Check if feature data exists for both regions
    if u not in PCA or v not in PCA or u not in LMEAN or v not in LMEAN:
        return 0.0

    # Get PCA endpoints and calculate vectors + lengths
    p1, p2 = PCA[u]; q1, q2 = PCA[v]
    vec1 = p2 - p1; vec2 = q2 - q1
    l1 = np.linalg.norm(vec1); l2 = np.linalg.norm(vec2)

    angleFac, lengthFac = 1.0, 1.0
    # Calculate angle and length penalties only if both lengths are significant
    if l1 > 1e-6 and l2 > 1e-6:
        v1_norm = vec1 / l1; v2_norm = vec2 / l2
        angleFac = angle_penalty(v1_norm, v2_norm, l1, l2)
        # Length penalty (favors merging regions with similar PCA lengths)
        lmin, lmax = (l1, l2) if l1 < l2 else (l2, l1)
        lengthFac = np.exp(-ETA * lmin / (lmax + 1e-9)) # Penalize large ratio differences

    # Color similarity penalty (based on distance between mean Lab colors)
    dLab = np.linalg.norm(LMEAN[u] - LMEAN[v])
    colFac = np.exp(-BETA * dLab * dLab)

    # Gradient penalty (discourages merging across strong edges)
    key = tuple(sorted((u, v)))
    g = GRAD.get(key, 0) # Default to 0 gradient if edge not found
    gradFac = np.exp(-GAMMA * g * g)

    # Combine factors
    return angleFac * lengthFac * colFac * gradFac

def cost(u, v):
    """Calculates merge cost (lower affinity -> higher cost)."""
    aff = affinity(u, v)
    # Use negative log of affinity; add small epsilon to avoid log(0)
    return -np.log(aff + 1e-12)


# Need access to lbl, lab in callback - make them available globally
# Note: Using globals like this is generally discouraged in larger applications,
# but retained here to match the original structure. Consider using classes
# or functools.partial if refactoring.
lbl = None
lab = None

def merge_cb(rag, src, dst):
    """Callback function executed when merging src into dst."""
    global lbl, lab # Declare usage of globals

    # Check if data exists for both source and destination nodes
    if src not in lab_sum or dst not in lab_sum or src not in lab_npix or dst not in lab_npix:
        logging.warning(f"Missing feature data during merge: src={src}, dst={dst}")
        return # Cannot proceed if data is missing

    # Update accumulated Lab sum and pixel count for the destination node
    lab_sum[dst] += lab_sum[src]
    lab_npix[dst] += lab_npix[src]

    # Update mean Lab color for the destination node
    LMEAN[dst] = lab_sum[dst] / (lab_npix[dst] + 1e-9) # Add epsilon for safety

    # --- Update PCA axis for the merged region (dst) ---
    try:
        # This part relies on global access to the label map (lbl) and Lab image (lab)
        if lbl is None or lab is None:
            raise NameError("Global 'lbl' or 'lab' not defined/available for merge_cb PCA update.")

        # Find all pixels belonging to the destination region *after* the merge
        # Note: This assumes lbl reflects the state *before* this specific merge completes conceptually.
        #       merge_hierarchical updates labels in-place. Need pixels for the *new* combined region.
        # A potentially safer way (if merge_hierarchical allows access to the underlying map):
        # pts_indices = np.where(rag.nodes[dst]['labels']) # If 'labels' attribute exists and is updated
        # For now, stick to original logic assuming lbl is accessible and reflects the pre-merge state relevant here.
        # We need pixels originally labeled src OR dst.
        # Let's find pixels currently labeled as dst in the *potentially updated* lbl map.
        # This might be slightly incorrect depending on merge_hierarchical's internal state management.

        # Find pixels labeled 'dst' *in the current state of lbl* (which merge_hierarchical modifies)
        pts_indices = np.where(lbl == dst) # This gets pixels NOW labeled as dst
        pts = lab[pts_indices] # Get their Lab values

        # Subsample if too many points to speed up SVD
        if pts.shape[0] > 3000:
            pts = pts[np.random.choice(pts.shape[0], 3000, replace=False)]

        # Calculate new PCA axis if enough points exist
        if pts.shape[0] >= 2:
            PCA[dst] = pca_axis(pts)
        elif dst in PCA: # Remove old PCA if not enough points
            del PCA[dst]

    except NameError as e:
        logging.error(f"Stopping merge due to missing global: {e}")
        raise # Re-raise to potentially stop the merge process if globals are critical
    except Exception as e:
        logging.error(f"PCA update error for region {dst} during merge: {e}")
        # Decide if we should continue without PCA update or stop
        if dst in PCA: del PCA[dst] # Remove potentially stale PCA


    # Remove data for the source node (it's now merged into dst)
    if src in lab_sum: del lab_sum[src]
    if src in lab_npix: del lab_npix[src]
    if src in LMEAN: del LMEAN[src]
    if src in PCA: del PCA[src]

def weight_cb(rag, src, dst, nbr):
    """Callback to calculate edge weight between merged node (dst) and its neighbor (nbr)."""
    # Check if necessary feature data exists for the destination and neighbor nodes
    if dst not in LMEAN or nbr not in LMEAN or dst not in PCA or nbr not in PCA:
         # Return infinite weight if data is missing, effectively preventing merge through this edge
         return {'weight': np.inf}

    # Calculate the new cost between the merged node and its neighbor
    new_cost = cost(dst, nbr)
    # Return finite cost or infinity if cost calculation failed
    return {'weight': new_cost if np.isfinite(new_cost) else np.inf}


# --- Run Original Segmentation ---
logging.info(f"Reading image: {IMG_PATH}")
try:
    rgb = io.imread(IMG_PATH)
    # Handle grayscale images by converting to RGB
    if rgb.ndim == 2:
        logging.warning("Input image is grayscale, converting to RGB.")
        rgb = color.gray2rgb(rgb)
    # Handle RGBA images by removing alpha channel
    elif rgb.shape[2] == 4:
         logging.warning("Input image has alpha channel, removing it.")
         rgb = rgb[:, :, :3]
    # Ensure image is uint8
    if rgb.dtype != np.uint8:
         logging.warning(f"Image data type is {rgb.dtype}, converting to uint8.")
         rgb = img_as_ubyte(rgb)

except FileNotFoundError:
    logging.error(f"Image file not found at {IMG_PATH}. Please check the path.")
    exit()
except Exception as e:
    logging.error(f"Error reading image {IMG_PATH}: {e}")
    exit()


H, W = rgb.shape[:2] # Get image dimensions
logging.info(f"Image dimensions: Height={H}, Width={W}")

# Convert RGB to Lab color space (float) and normalize
# Normalization might need adjustment based on Lab range characteristics
# Original normalization: L range [-1, 1], a/b range approx [-1, 1]
lab_raw = color.rgb2lab(rgb).astype(float)
lab = np.zeros_like(lab_raw)
lab[..., 0] = (lab_raw[..., 0] - 50) / 50 # Center L around 0, scale approx to [-1, 1]
lab[..., 1:] = lab_raw[..., 1:] / 110 # Scale a/b channels (approx range +/- 110)
logging.info("Applied original Lab normalization.")


# Convert RGB to float [0, 1] for SLIC and blurring
rgb_float = img_as_float(rgb)

# Gaussian blurring before SLIC (optional, can help smooth noise)
# sigma=0 means no blurring, sigma=1 is mild blurring
sigma_blur = 1.0 # Keep original sigma
if sigma_blur > 0:
     rgb_blurred = filters.gaussian(rgb_float, sigma=sigma_blur, channel_axis=-1, preserve_range=True)
     logging.info(f"Applied Gaussian blur with sigma={sigma_blur}.")
else:
     rgb_blurred = rgb_float
     logging.info("Skipping Gaussian blur.")


# --- SLIC Segmentation ---
# start_label=0 ensures labels start from 0 upwards
logging.info(f"Running SLIC: n_segments={N_SEGMENTS}, compactness={COMPACTNESS}...")
lbl = segmentation.slic(rgb_blurred, n_segments=N_SEGMENTS, compactness=COMPACTNESS, start_label=0, channel_axis=-1, enforce_connectivity=True)
# Relabel sequentially to ensure contiguous labels from 0 to N-1
lbl = segmentation.relabel_sequential(lbl)[0] # Returns labels and number of labels
num_slic_regions = lbl.max() + 1
logging.info(f"SLIC generated {num_slic_regions} initial regions (relabelled).")

# --- Calculate Initial Features for each SLIC region ---
logging.info("Calculating initial features (Mean Lab, PCA) for SLIC regions...")
# Clear global dictionaries before recalculating
LMEAN.clear(); PCA.clear(); lab_sum.clear(); lab_npix.clear(); GRAD.clear()

unique_slic_labels = np.unique(lbl)
for l in unique_slic_labels:
    if l < 0: continue # Should not happen after relabel_sequential with start_label=0
    pix_mask = lbl == l
    # Ensure the region actually has pixels (can happen with small segments)
    if not np.any(pix_mask): continue

    pix_lab = lab[pix_mask] # Get Lab values for pixels in region l
    # Ensure there are pixels before calculating mean/sum
    if pix_lab.shape[0] == 0: continue

    LMEAN[l] = pix_lab.mean(0)
    lab_sum[l] = pix_lab.sum(0)
    lab_npix[l] = pix_lab.shape[0]
    # Calculate PCA only if enough points exist
    if lab_npix[l] >= 2:
        PCA[l] = pca_axis(pix_lab)
    # else: PCA[l] remains unset for this label

logging.info(f"Calculated initial features for {len(LMEAN)} regions.")


# --- Calculate Boundary Gradients using Canny ---
logging.info("Calculating boundary gradients using Canny...")
gray = color.rgb2gray(rgb_float)
# Adjust Canny sigma if needed; higher sigma detects larger scale edges
canny_sigma = 1.0
Gmag = canny(gray, sigma=canny_sigma).astype(float)
logging.info(f"Canny edge detection completed (sigma={canny_sigma}).")

# Build initial Region Adjacency Graph (RAG) based on SLIC labels
# mode='distance' uses average Euclidean distance in color space (not used by custom cost)
rag_initial = graph.rag_mean_color(rgb_float, lbl, mode='distance')
logging.info(f"Initial RAG built with {rag_initial.number_of_nodes()} nodes and {rag_initial.number_of_edges()} edges.")


# Calculate mean gradient across boundaries between adjacent SLIC regions
edge_count = 0
for u, v in rag_initial.edges:
    # Ensure features exist for both nodes (might not if a region was empty)
    if u not in LMEAN or v not in LMEAN: continue

    # Find boundary pixels between region u and region v
    mask_u = lbl == u
    mask_v = lbl == v
    # Dilate mask_u and find intersection with mask_v to get boundary pixels in v adjacent to u
    # Use a simple connectivity structure (e.g., 4-connectivity)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]]) # Cross structure (4-connectivity)
    # Alternative: structure=np.ones((3,3)) for 8-connectivity
    boundary_v = binary_dilation(mask_u, structure=structure, border_value=0) & mask_v

    # Get coordinates of boundary pixels and calculate mean Canny gradient magnitude
    coords = np.column_stack(np.where(boundary_v))
    mean_grad = 0.0
    if coords.size > 0:
        # Ensure coordinates are within image bounds (should be, but safety check)
        coords = coords[(coords[:, 0] < H) & (coords[:, 1] < W)]
        if coords.size > 0:
            mean_grad = Gmag[coords[:, 0], coords[:, 1]].mean()

    # Store gradient in GRAD dictionary (use sorted tuple as key)
    GRAD[tuple(sorted((u, v)))] = mean_grad
    edge_count += 1

logging.info(f"Processed gradients for {edge_count} edges between valid regions.")


# --- Build RAG with Custom Costs for Hierarchical Merging ---
logging.info("Building RAG with custom costs based on affinity...")
rag = rag_initial.copy() # Start with the initial RAG structure
edges_removed_count = 0
edges_updated_count = 0

# Iterate through edges and assign custom cost based on affinity
for u, v, edge_data in rag.edges(data=True):
    # Check if features exist for both nodes involved in the edge
    if u in LMEAN and v in LMEAN and u in PCA and v in PCA:
        edge_cost = cost(u, v) # Calculate custom cost
        # Assign weight only if cost is finite
        if np.isfinite(edge_cost):
            edge_data['weight'] = edge_cost
            edges_updated_count += 1
        else:
            # Assign infinite weight to prevent merging if cost is invalid
            edge_data['weight'] = np.inf
            edges_removed_count += 1 # Count as effectively removed for merging
    else:
        # Assign infinite weight if features are missing
        edge_data['weight'] = np.inf
        edges_removed_count += 1

logging.info(f"Custom RAG prepared: {rag.number_of_nodes()} nodes, {rag.number_of_edges()} edges.")
if edges_removed_count > 0:
     logging.info(f"Assigned infinite weight (preventing merge) to {edges_removed_count} edges due to missing features or non-finite cost.")


# --- Hierarchical Merging ---
logging.info(f"Starting hierarchical merging with Thresh={THRESH}...")
# merge_hierarchical modifies lbl and rag in place
# rag_copy=False, in_place_merge=True are important for efficiency and callbacks
lbl2 = graph.merge_hierarchical(
    lbl, # Initial label map (will be modified in place)
    rag, # RAG with custom weights (will be modified in place)
    thresh=THRESH, # Threshold for merging (merge if cost < thresh)
    rag_copy=False, # Do not copy RAG
    in_place_merge=True, # Merge nodes in the original RAG
    merge_func=merge_cb, # Function called after nodes are merged
    weight_func=weight_cb # Function called to calculate new edge weights
)
num_final_regions = len(np.unique(lbl2))
logging.info(f"Hierarchical merging complete: {num_slic_regions} initial regions -> {num_final_regions} final regions.")


# --- Show Original Segmentation Result ---
try:
    logging.info("Generating segmentation visualization plot...")
    fig_seg, ax_seg = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Left plot: Initial SLIC segmentation
    ax_seg[0].imshow(segmentation.mark_boundaries(rgb_float, lbl, color=(1, 0, 0), mode='thick'))
    ax_seg[0].set_title(f'SLIC ({num_slic_regions} regions)')
    ax_seg[0].axis('off')

    # Right plot: Final segmentation after merging
    ax_seg[1].imshow(segmentation.mark_boundaries(rgb_float, lbl2, color=(0, 1, 0), mode='thick'))
    ax_seg[1].set_title(f'After Merge ({num_final_regions} regions, Thresh={THRESH})')
    ax_seg[1].axis('off')

    plt.tight_layout()
    # Save the plot to the data directory
    plot_path = PREVIEW_DIR / f"{IMG_PATH.stem}_segmentation_result_ORIGINAL.png"
    plt.savefig(plot_path)
    logging.info(f"Segmentation visualization saved to {plot_path}")
    plt.show() # Display the plot
    plt.close(fig_seg) # Close the figure to free memory

except Exception as e:
    logging.error(f"Failed to generate or save segmentation plot: {e}")
    # Optionally print traceback for detailed debugging
    # logging.error(traceback.format_exc())


#%%
# =====================================================================
# NEW Part: Polygon Generation and Level-Based JSON Export
# Using the lbl2 generated above with the user's preferred parameters
# =====================================================================

# ————————————————————————————————————————————————————————————————
# MAIN - Level-Based JSON Generation
# ————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    # H, W should be available from segmentation part

    # ---- ❶ Generate Polygons using robust function ----
    logging.info("Generating polygons from final merged labels (using robust function)...")
    rid2poly_initial = {} # Initialize dictionary
    try:
        # --- *** THIS BLOCK IS NOW UNCOMMENTED *** ---
        # Use low SIMPLIFY_PX for vector export
        # Pass the final label map lbl2 to the function
        rid2poly_initial = dict(polygons_from_labels(lbl2, simplify_px=SIMPLIFY_PX, min_ring_area=MIN_RING_AREA))

        # --- Optional: Remove region 0 if it's known background noise ---
        # Be careful: If sky/grass is region 0, this will remove it.
        remove_region_zero = False # Set to True only if needed
        if remove_region_zero and 0 in rid2poly_initial:
             logging.info("Explicitly removing region 0 before forest building.")
             del rid2poly_initial[0]
        # --- *** End of formerly commented block *** ---

        num_initial_polys = len(rid2poly_initial)
        logging.info(f"Generated {num_initial_polys} initial polygons for forest building.")
        if num_initial_polys == 0:
             logging.warning("No polygons were generated from the labels. Check segmentation (lbl2) and polygon generation filters/logic.")
             # Exit gracefully if no polygons were created
             exit()


    except NameError as ne:
        # This error should not happen now, but kept for safety
        logging.error(f"NameError during polygon generation: {ne}. Is 'polygons_from_labels' defined?")
        logging.error(traceback.format_exc())
        exit()
    except Exception as e:
        logging.error(f"Error during polygon generation: {e}")
        logging.error(traceback.format_exc())
        exit()

    # ---- ❷ Build Containment Forest ----
    logging.info("Building region containment forest...")
    try:
        roots, max_depth_calculated = build_region_forest(rid2poly_initial)
        logging.info(f"Forest built. Found {len(roots)} roots. Max depth: {max_depth_calculated}.")
    except NameError as ne:
        logging.error(f"NameError during forest building: {ne}. Is 'build_region_forest' defined?")
        logging.error(traceback.format_exc())
        exit()
    except Exception as e:
        logging.error(f"Error building region forest: {e}")
        logging.error(traceback.format_exc())
        exit()

    # ---- ❸ Export LabelMe JSONs by Level ----
    logging.info("Exporting LabelMe JSON files per depth level...")
    try:
        # Pass the original image path object, dimensions, roots, and max depth
        export_labelme_by_level(IMG_PATH, H, W, roots, max_depth_calculated)
    except NameError as ne:
        logging.error(f"NameError during JSON export: {ne}. Are 'export_labelme_by_level', 'inner_chords', or 'Region' defined?")
        logging.error(traceback.format_exc())
        exit()
    except Exception as e:
        logging.error(f"Error exporting LabelMe JSONs: {e}")
        logging.error(traceback.format_exc())
        exit()

    # ---- ❹ Final Visual Check (Load generated JSONs) ----

    # --- Part A: Generate Layer-by-Layer Previews ---
    logging.info("Generating layer-by-layer preview plots...")
    # Find all exported JSON files for this image stem in the image's parent directory
    json_files = sorted(list(IMG_PATH.parent.glob(f"{IMG_PATH.stem}_labelme_L*.json")))

    if not json_files:
        logging.warning("No JSON files found. Skipping all preview generation.")
    else:
        for json_file in json_files:
            try:
                level = int(json_file.stem.split('_L')[-1])
                lm_data = json.loads(json_file.read_text())

                fig_layer, ax_layer = plt.subplots(figsize=(10, 10))
                ax_layer.imshow(rgb) # Show original image
                ax_layer.set_title(f"LabelMe Preview - Level {level}")
                ax_layer.axis('off')

                # Use a consistent color scheme for each layer plot for simplicity here
                # (e.g., blue for polygons, green for lines)
                poly_color = 'blue'
                line_color = 'lime'

                logging.info(f"Plotting shapes from {json_file.name} (Level {level})")
                for sh in lm_data.get("shapes", []): # Use .get for safety
                    pts = np.asarray(sh.get("points", []))
                    shape_type = sh.get("shape_type")

                    if shape_type == "polygon" and pts.shape[0] >= 3:
                        ax_layer.fill(pts[:, 0], pts[:, 1],
                                        facecolor=poly_color,
                                        alpha=0.4, # Semi-transparent fill
                                        edgecolor=poly_color, # Solid edge
                                        linewidth=1.5)
                    elif shape_type == "line" and pts.shape[0] == 2:
                        ax_layer.plot(pts[:, 0], pts[:, 1],
                                        color=line_color,
                                        linestyle='-',
                                        linewidth=2.0, # Make lines slightly thicker
                                        alpha=0.9) # Mostly opaque

                # Save the individual layer plot
                preview_path_layer = PREVIEW_DIR / f"{IMG_PATH.stem}_labelme_preview_L{level}.png"
                plt.tight_layout()
                plt.savefig(preview_path_layer)
                logging.info(f"Layer preview saved to {preview_path_layer}")
                plt.close(fig_layer) # Close figure to free memory

            except ValueError:
                 logging.error(f"Could not parse level number from filename: {json_file.name}")
            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON from file: {json_file.name}")
            except Exception as e:
                 logging.error(f"Failed to load/parse/plot shapes from {json_file.name}: {e}", exc_info=True)

    # --- Part B: Generate Improved Combined Preview (Color by Region ID) ---
    logging.info("Generating combined preview plot (colored by Region ID)...")
    if not json_files:
        logging.warning("No JSON files found for combined preview.")
    else:
        try:
            fig_combined, ax_combined = plt.subplots(figsize=(11, 11)) # Slightly larger
            ax_combined.imshow(rgb)
            ax_combined.set_title("LabelMe Preview (All Levels - Colored by Region ID)")
            ax_combined.axis('off')

            # Gather all shapes and determine unique region IDs
            all_shapes = []
            region_ids = set()
            for json_file in json_files:
                try:
                    lm_data = json.loads(json_file.read_text())
                    shapes = lm_data.get("shapes", [])
                    all_shapes.extend(shapes)
                    for sh in shapes:
                        label = sh.get("label", "")
                        # Extract region ID (assuming format like "region-ID" or "direction-ID")
                        parts = label.split('-')
                        if len(parts) > 1:
                            region_ids.add(parts[-1]) # Add the ID part
                except Exception as e:
                    logging.warning(f"Could not process shapes from {json_file.name} for combined plot: {e}")

            # Create a color map for region IDs
            region_id_list = sorted(list(region_ids))
            cmap = plt.colormaps.get_cmap('tab20') # Use a colormap with distinct colors
            region_colors = {rid: cmap(i % cmap.N) for i, rid in enumerate(region_id_list)}
            logging.info(f"Found {len(region_colors)} unique region IDs for coloring.")

            # Plot all shapes using region ID colors
            for sh in all_shapes:
                pts = np.asarray(sh.get("points", []))
                shape_type = sh.get("shape_type")
                label = sh.get("label", "")
                parts = label.split('-')
                region_id = parts[-1] if len(parts) > 1 else None

                if region_id and region_id in region_colors:
                    color = region_colors[region_id]
                    if shape_type == "polygon" and pts.shape[0] >= 3:
                        ax_combined.fill(pts[:, 0], pts[:, 1],
                                         facecolor=list(color[:3]) + [0.45], # RGB + Alpha (slightly more opaque)
                                         edgecolor=color,
                                         linewidth=1.0) # Thinner edge for filled polygon
                    elif shape_type == "line" and pts.shape[0] == 2:
                         # Use same color as polygon, but solid and thicker line
                        ax_combined.plot(pts[:, 0], pts[:, 1],
                                         color=color,
                                         linestyle='-',
                                         linewidth=2.0, # Thicker line
                                         alpha=1.0)    # Opaque line
                else:
                    # Fallback for shapes without recognizable region ID in label
                    if shape_type == "polygon" and pts.shape[0] >= 3:
                        ax_combined.fill(pts[:, 0], pts[:, 1], facecolor='gray', alpha=0.3, edgecolor='black', linewidth=0.5)
                    elif shape_type == "line" and pts.shape[0] == 2:
                        ax_combined.plot(pts[:, 0], pts[:, 1], color='gray', linestyle=':', linewidth=1.0, alpha=0.7)


            # Save the combined plot
            preview_path_combined = PREVIEW_DIR / f"{IMG_PATH.stem}_labelme_preview_COMBINED_BY_REGION.png"
            plt.tight_layout()
            plt.savefig(preview_path_combined)
            logging.info(f"Combined preview (by Region ID) saved to {preview_path_combined}")
            # plt.show() # Optionally show combined plot interactively
            plt.close(fig_combined) # Close figure

        except Exception as e:
            logging.error(f"Failed to generate combined LabelMe preview plot: {e}", exc_info=True)


    logging.info("Segmentation and JSON export script finished.")

# --- End Main Execution Block ---
"""
    # ---- ❹ Final Visual Check (Load generated JSONs) ----
    logging.info("Generating final preview plot from exported JSON files...")
    try:
        fig_preview, ax_preview = plt.subplots(figsize=(10, 10)) # Slightly larger figure
        # Display the original image as background
        ax_preview.imshow(rgb)
        ax_preview.set_title("LabelMe Preview (All Levels - Vector Polygons)")

        num_depths = max_depth_calculated + 1
        # Define colormap - use tab20 for more distinct colors if many levels, otherwise viridis
        # Use matplotlib.colormaps registry instead of deprecated get_cmap
        if 0 < num_depths <= 20:
            cmap_func = plt.colormaps.get_cmap('tab20')
            colors = [cmap_func(i % cmap_func.N) for i in range(num_depths)]
        elif num_depths > 20:
             cmap_func = plt.colormaps.get_cmap('tab20') # Fallback to repeating tab20
             colors = [cmap_func(i % cmap_func.N) for i in range(num_depths)]
             logging.warning(f"More than 20 depth levels ({num_depths}), colors will repeat.")
        else: # Only level 0 or no levels
             colors = ['red'] # Default color if no depth or only L0


        # Find all exported JSON files for this image stem
        json_files = sorted(list(DATA_DIR.glob(f"{IMG_PATH.stem}_labelme_L*.json")))
        logging.info(f"Found {len(json_files)} JSON files to visualize.")

        if not json_files:
            logging.warning("No JSON files found for preview. Skipping preview generation.")
        else:
            # Plot shapes from each JSON file
            for json_file in json_files:
                try:
                    level = int(json_file.stem.split('_L')[-1])
                    lm_data = json.loads(json_file.read_text())

                    # Determine colors based on level index
                    level_idx = level % len(colors)
                    level_color_poly = colors[level_idx]
                    level_color_line = 'lime' # Keep lines visually distinct

                    logging.info(f"Plotting shapes from {json_file.name} (Level {level}) with color {level_color_poly}")

                    # Iterate through shapes in the JSON
                    for sh in lm_data["shapes"]:
                        pts = np.asarray(sh["points"])
                        if sh["shape_type"] == "polygon" and pts.shape[0] >= 3:
                            # Plot filled polygon with transparency
                            ax_preview.fill(pts[:, 0], pts[:, 1],
                                            facecolor=list(level_color_poly[:3]) + [0.4], # RGB + Alpha
                                            edgecolor=level_color_poly,
                                            linewidth=1.5) # Slightly thicker edge
                        elif sh["shape_type"] == "line" and pts.shape[0] == 2:
                            # Plot direction lines
                            ax_preview.plot(pts[:, 0], pts[:, 1],
                                            color=level_color_line,
                                            linestyle='-',
                                            linewidth=1.5, # Slightly thicker line
                                            alpha=0.8) # Slightly transparent

                except ValueError:
                     logging.error(f"Could not parse level number from filename: {json_file.name}")
                except json.JSONDecodeError:
                     logging.error(f"Failed to decode JSON from file: {json_file.name}")
                except Exception as e:
                     logging.error(f"Failed to load or parse shapes from {json_file.name}: {e}")


            # Finalize and save the preview plot
            ax_preview.axis('off') # Hide axes
            plt.tight_layout()
            preview_path = DATA_DIR / f"{IMG_PATH.stem}_labelme_preview_LEVELS.png"
            plt.savefig(preview_path)
            logging.info(f"Level-based LabelMe preview saved to {preview_path}")
            plt.show() # Display the plot
            plt.close(fig_preview) # Close figure

    except Exception as e:
        logging.error(f"Failed to generate LabelMe preview plot: {e}")
        logging.error(traceback.format_exc())


    logging.info("Segmentation and JSON export script finished.")
"""
# --- End Main Execution Block ---