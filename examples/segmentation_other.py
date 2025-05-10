#%%
# Generates LabelMe JSON files for K-Means, SLIC, and simulated Object Detection segmentations.
# Output JSONs contain region polygons AND automatically generated direction lines.
# Labels are formatted as region-ID and direction-ID.
# K-Means is run with 6 clusters in RGB space for comparison purposes.
# Ensures only the largest component for each region ID is kept.
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, color, measure, img_as_float, img_as_ubyte
from sklearn.cluster import KMeans
try:
    from shapely.geometry import Polygon, MultiPolygon, LineString, Point
    from shapely.ops import unary_union
except ImportError:
     print("ERROR: Shapely library not found. Please install it (`pip install shapely`)")
     exit()
import base64
import json
from pathlib import Path
import logging
import traceback
import random # Import random module for seeding
from typing import List, Dict, Tuple, Any # Added for type hinting

# --- Configuration ---
# Input Image Path (Modify as needed)
IMG_FILE = "data/2color.jpg" 

# Output Directory for JSON files
OUTPUT_DIR = Path("data") 

# Target number of regions/clusters for SLIC
SLIC_TARGET_REGIONS = 4 

# --- Clusters for K-Means comparison (using RGB space) ---
KMEANS_TARGET_REGIONS = 4 # Use 6 clusters for K-Means as requested

# Number of clusters for simulating object detection (e.g., background + 2-3 objects)
OBJDET_SIM_CLUSTERS = 2 

# Parameters for Polygon Generation
SIMPLIFY_PX = 0.5       # Simplification tolerance (pixels).
MIN_POLYGON_AREA = 50 # Minimum area (pixels^2) for a polygon to be included.

# Parameters for Direction Line (Chord) Generation
N_DIRS = 3 # how many direction chords per region
MIN_FRAC = .25 # min length of chord relative to max possible length
MAX_FRAC = .45 # max length of chord relative to max possible length
FAN_DEG = 75 # angular spread (± deg) around PCA axis for chords

# --- End Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Function: Inner Chords (Copied from segmentation script) ---
# Note: This version doesn't use 'forbidden_polys' as we process isolated regions here.
def inner_chords(poly: Polygon, rng: np.random.Generator, n_dirs=N_DIRS, min_frac=MIN_FRAC, max_frac=MAX_FRAC, fan_deg=FAN_DEG):
    """Generates inner direction lines (chords) for a given polygon."""
    if poly.is_empty or not poly.is_valid or poly.area < 1e-6: return []
    current_poly = poly if poly.is_valid else poly.buffer(0)
    if not current_poly.is_valid or current_poly.is_empty:
        logging.warning(f"Skipping inner_chords for invalid/empty polygon (Area: {poly.area:.2f})")
        return []

    try: repr_point = current_poly.representative_point(); Cx, Cy = repr_point.x, repr_point.y; center_pt = Point(Cx, Cy)
    except Exception:
        try: centroid = current_poly.centroid; Cx, Cy = centroid.x, centroid.y; center_pt = centroid
        except Exception: logging.error(f"Could not get center point for polygon area {poly.area:.1f}. Cannot generate chords."); return []

    poly_id_str = f"poly_area_{poly.area:.1f}"
    axis = None # Initialize axis
    try: 
        xy = np.asarray(current_poly.exterior.coords) - (Cx, Cy)
        if xy.shape[0] > 2:
            _, _, Vt = np.linalg.svd(xy[:-1], full_matrices=False)
            axis = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-9)
        else: logging.warning(f"[{poly_id_str}] Not enough points for SVD.")
    except Exception as e: logging.warning(f"[{poly_id_str}] SVD failed for PCA axis: {e}. Using random axis.")
    
    if axis is None: # Fallback if SVD failed or not enough points
        axis = rng.random(2) * 2 - 1
        axis = axis / (np.linalg.norm(axis) + 1e-9)

    # --- Primary Method: PCA-based random chords ---
    segs = []; tries = 0; max_tries = n_dirs * 200
    while len(segs) < n_dirs and tries < max_tries:
        tries += 1
        angle_rad = np.deg2rad(rng.uniform(-fan_deg, fan_deg)); rot_matrix = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]]); v = rot_matrix @ axis
        bounds = current_poly.bounds
        if not (len(bounds)==4 and all(isinstance(b,float) for b in bounds)): continue
        diag_len = np.sqrt((bounds[2]-bounds[0])**2 + (bounds[3]-bounds[1])**2); far = diag_len*1.5 + 10
        line = LineString([(Cx - far * v[0], Cy - far * v[1]), (Cx + far * v[0], Cy + far * v[1])])
        try: chord = line.intersection(current_poly)
        except Exception: continue
        if chord.is_empty or not isinstance(chord, LineString) or len(chord.coords)<2: continue
        c0, c1 = map(np.asarray, chord.coords); full_len = np.linalg.norm(c1 - c0)
        if full_len < 1e-3: continue
        min_len_abs = min_frac * full_len; max_len_abs = max_frac * full_len
        if max_len_abs <= min_len_abs or min_len_abs < 0: continue
        target_len = rng.uniform(min_len_abs, max_len_abs); start_offset = rng.uniform(0, full_len - target_len) if full_len > target_len else 0
        end_offset = start_offset + target_len; unit_vec = (c1 - c0) / (full_len + 1e-9)
        pad = min(0.5, target_len * 0.01, full_len * 0.005)
        p0 = c0 + unit_vec * (start_offset + pad); p1 = c0 + unit_vec * (end_offset - pad)
        if np.linalg.norm(p1 - p0) < 1e-3: continue
        seg_line = LineString([tuple(p0), tuple(p1)])
        try:
            is_contained = current_poly.buffer(-1e-6).contains(seg_line)
            if not is_contained: continue
            # No forbidden check needed here as we process isolated regions
            segs.append([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]])
        except Exception: continue

    # --- Fallback Method: Center to random boundary points ---
    fallback_tries = 0
    if len(segs) < n_dirs:
        logging.warning(f"[{poly_id_str}] Primary method failed ({len(segs)}/{n_dirs}). Trying fallback.")
        needed = n_dirs - len(segs); max_fallback_tries = needed * 50
        if current_poly.exterior is None or len(current_poly.exterior.coords) < 2:
             logging.error(f"[{poly_id_str}] Cannot use fallback: Polygon exterior invalid.")
        else:
            boundary_len = current_poly.exterior.length
            while len(segs) < n_dirs and fallback_tries < max_fallback_tries:
                fallback_tries += 1
                try:
                    random_dist = rng.uniform(0, boundary_len); boundary_pt = current_poly.exterior.interpolate(random_dist)
                    center_to_boundary = LineString([center_pt, boundary_pt])
                    if center_to_boundary.length < 1e-6: continue
                    target_fallback_len = max(0.5, center_to_boundary.length * 0.9) 
                    seg_line = LineString([center_pt, center_to_boundary.interpolate(target_fallback_len)])
                    # No forbidden check needed
                    if not current_poly.buffer(-1e-6).contains(seg_line.representative_point()): continue
                    p0_fb, p1_fb = seg_line.coords
                    segs.append([[float(p0_fb[0]), float(p0_fb[1])], [float(p1_fb[0]), float(p1_fb[1])]])
                except Exception: continue

    # --- Final Logging ---
    if len(segs) < n_dirs: logging.error(f"[{poly_id_str}] FAILED to generate {n_dirs} lines. Got {len(segs)}.")
    elif fallback_tries > 0 : logging.info(f"[{poly_id_str}] Generated {len(segs)}/{n_dirs} lines (used fallback).")
    # else: logging.info(f"[{poly_id_str}] Generated {len(segs)}/{n_dirs} lines (primary method).") # Can be noisy

    return segs

# --- Helper Function: Generate Polygons from Label Map (Modified) ---
def polygons_from_label_map(label_img, simplify_px: float = SIMPLIFY_PX, min_area: float = MIN_POLYGON_AREA, ignore_largest_region=False):
    """
    Generates Shapely Polygons for each unique label in a label map.
    If a label corresponds to multiple disconnected polygons (MultiPolygon), 
    only the largest one is kept.
    Optionally ignores the overall largest region.
    """
    polygons = {}
    unique_labels, counts = np.unique(label_img, return_counts=True)
    logging.info(f"  Extracting polygons for {len(unique_labels)} unique labels...")
    
    label_to_ignore = -1
    if ignore_largest_region and len(counts) > 0:
        largest_region_idx = np.argmax(counts)
        label_to_ignore = unique_labels[largest_region_idx]
        logging.info(f"  Ignoring largest region (label {label_to_ignore}) for polygon extraction.")
        
    for region_id in unique_labels:
        if region_id < 0 or (ignore_largest_region and region_id == label_to_ignore): 
            continue
            
        mask = label_img == region_id
        if not np.any(mask): continue

        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        try:
            contours = measure.find_contours(padded_mask.astype(float), 0.5)
            contours = [c - 1 for c in contours] # Adjust back to original coords
        except Exception as e: logging.error(f"  find_contours failed for region {region_id}: {e}"); continue
        if not contours: continue

        region_polygons = []
        for contour in contours:
            if len(contour) < 4: continue
            try: poly = Polygon(np.fliplr(contour)).buffer(0)
            except Exception: continue 
            if poly.is_valid and not poly.is_empty: region_polygons.append(poly)
        if not region_polygons: continue

        try: merged_geom = unary_union(region_polygons).buffer(0)
        except Exception as e: logging.error(f"  Unary union failed for region {region_id}: {e}"); continue

        # --- MODIFIED: Select largest component if MultiPolygon ---
        geom_to_process = None
        if isinstance(merged_geom, Polygon):
            geom_to_process = merged_geom
        elif isinstance(merged_geom, MultiPolygon):
            valid_parts = [p for p in merged_geom.geoms if p.is_valid and p.area >= min_area] # Apply min_area here too
            if valid_parts:
                geom_to_process = max(valid_parts, key=lambda p: p.area)
                logging.debug(f"  Region {region_id}: Kept largest component (Area: {geom_to_process.area:.1f}) from MultiPolygon.")
            else:
                 logging.debug(f"  Region {region_id}: No valid components found in MultiPolygon after area filter.")
        # --- END MODIFIED ---

        if geom_to_process and geom_to_process.is_valid and not geom_to_process.is_empty:
            try:
                # Simplify the selected (largest) polygon
                simplified_geom = geom_to_process.simplify(simplify_px, preserve_topology=True).buffer(0)
                if simplified_geom.is_valid and not simplified_geom.is_empty and simplified_geom.area >= min_area:
                    polygons[region_id] = simplified_geom # Store the single largest, simplified polygon
                else:
                     logging.debug(f"  Region {region_id}: Largest component failed simplification or area filter.")
            except Exception as e:
                logging.warning(f"  Simplification failed for region {region_id}: {e}")
        else:
             logging.debug(f"  Region {region_id}: No valid single component found after merge/selection.")

    logging.info(f"  Finished polygon extraction. Found valid polygons for {len(polygons)} regions.")
    return polygons

# --- Helper Function: Create LabelMe JSON (Includes Directions) ---
def create_labelme_json(polygons_dict, img_path_obj: Path, H: int, W: int, method_name: str):
    """ Creates the LabelMe JSON structure including direction lines. """
    shapes = []
    # Seed RNG based on method name hash (use abs value)
    rng = np.random.default_rng(seed=abs(hash(method_name))) 

    for region_id, poly in polygons_dict.items(): # Now iterates through single Polygons
        if not isinstance(poly, Polygon): continue
        if poly.exterior is None or len(poly.exterior.coords) < 3: continue
        
        # Add Polygon Shape with label "region-X"
        points = np.array(poly.exterior.coords[:-1]).tolist() 
        shape_data = { "label": f"region-{region_id}", "points": points, "group_id": None, "shape_type": "polygon", "flags": {} }
        shapes.append(shape_data)

        # Generate and add direction lines with label "direction-X"
        region_rng = np.random.default_rng(seed=int(region_id)) 
        direction_lines = inner_chords(poly, region_rng) 
        for seg_coords in direction_lines:
             line_shape = { "label": f"direction-{region_id}", "points": seg_coords, "group_id": None, "shape_type": "line", "flags": {} }
             shapes.append(line_shape)
                 
    # Embed image data
    img_b64 = "" # Initialize
    try: 
        img_bytes = img_path_obj.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        logging.info(f"  Read and encoded image data for {img_path_obj.name}.")
    except Exception as e: 
        logging.error(f"  Failed to read/encode image file {img_path_obj}: {e}")

    labelme_data = { "version": "5.2.1", "flags": {}, "shapes": shapes, "imagePath": img_path_obj.name, "imageData": img_b64, "imageHeight": H, "imageWidth": W }
    return labelme_data

# --- Main Processing ---
img_path = Path(IMG_FILE)
if not img_path.exists(): logging.error(f"Input image not found: {img_path}"); exit()

logging.info(f"Processing image: {img_path}")
try:
    img_rgb_uint8 = io.imread(img_path)
    if img_rgb_uint8.ndim == 2: img_rgb_uint8 = color.gray2rgb(img_rgb_uint8)
    if img_rgb_uint8.shape[2] == 4: img_rgb_uint8 = img_rgb_uint8[:,:,:3]
    img_rgb_uint8 = img_as_ubyte(img_rgb_uint8)
    img_rgb_float = img_as_float(img_rgb_uint8)
    H, W = img_rgb_float.shape[:2]
    logging.info(f"Image dimensions: H={H}, W={W}")
    img_lab = color.rgb2lab(img_rgb_float) # Use Lab space for ObjDet Sim
except Exception as e: logging.error(f"Error loading/converting image: {e}"); exit()

# --- 1. K-Means Segmentation (Using RGB space, KMEANS_TARGET_REGIONS clusters) ---
logging.info(f"\n--- Running K-Means (k={KMEANS_TARGET_REGIONS}, RGB space) ---") 
try:
    # --- Use RGB pixels ---
    pixels_rgb = img_rgb_float.reshape(-1, 3) 
    kmeans = KMeans(n_clusters=KMEANS_TARGET_REGIONS, random_state=42, n_init=10) 
    kmeans.fit(pixels_rgb)
    # --- END RGB ---
    kmeans_labels = kmeans.labels_.reshape(H, W)
    # Keep only largest component for each label
    kmeans_polygons = polygons_from_label_map(kmeans_labels, min_area=MIN_POLYGON_AREA) 
    kmeans_json_data = create_labelme_json(kmeans_polygons, img_path, H, W, "kmeans") 
    kmeans_json_filename = OUTPUT_DIR / f"{img_path.stem}_kmeans_L0.json" 
    with open(kmeans_json_filename, 'w') as f: json.dump(kmeans_json_data, f, indent=2)
    logging.info(f"✓ K-Means JSON saved to: {kmeans_json_filename}")
except Exception as e: logging.error(f"K-Means processing failed: {e}", exc_info=True)

# --- 2. SLIC Segmentation ---
logging.info(f"\n--- Running SLIC (target_regions={SLIC_TARGET_REGIONS}) ---") 
try:
    slic_segments = segmentation.slic(img_rgb_float, n_segments=SLIC_TARGET_REGIONS, compactness=10, sigma=1, start_label=0, channel_axis=-1)
    num_slic_generated = len(np.unique(slic_segments)); logging.info(f"  SLIC generated {num_slic_generated} actual segments.")
    # Keep only largest component for each label (though SLIC usually produces connected segments)
    slic_polygons = polygons_from_label_map(slic_segments, min_area=MIN_POLYGON_AREA) 
    slic_json_data = create_labelme_json(slic_polygons, img_path, H, W, "slic") 
    slic_json_filename = OUTPUT_DIR / f"{img_path.stem}_slic_L0.json" 
    with open(slic_json_filename, 'w') as f: json.dump(slic_json_data, f, indent=2)
    logging.info(f"✓ SLIC JSON saved to: {slic_json_filename}")
except Exception as e: logging.error(f"SLIC processing failed: {e}", exc_info=True)

# --- 3. Simulated Object Detection ---
logging.info(f"\n--- Running Simulated Object Detection (k={OBJDET_SIM_CLUSTERS}) ---")
try:
    # Use K-Means on Lab with few clusters for ObjDet simulation
    pixels_lab_obj = img_lab.reshape(-1, 3) 
    kmeans_obj = KMeans(n_clusters=OBJDET_SIM_CLUSTERS, random_state=42, n_init=10)
    kmeans_obj.fit(pixels_lab_obj)
    objdet_labels = kmeans_obj.labels_.reshape(H, W)
    # Keep only largest component for each label, ignoring the overall largest label
    objdet_polygons = polygons_from_label_map(objdet_labels, min_area=MIN_POLYGON_AREA, ignore_largest_region=True) 
    objdet_json_data = create_labelme_json(objdet_polygons, img_path, H, W, "objdet") 
    objdet_json_filename = OUTPUT_DIR / f"{img_path.stem}_objdet_L0.json" 
    with open(objdet_json_filename, 'w') as f: json.dump(objdet_json_data, f, indent=2)
    logging.info(f"✓ Simulated Object Detection JSON saved to: {objdet_json_filename}")
except Exception as e: logging.error(f"Simulated Object Detection processing failed: {e}", exc_info=True)

logging.info("\n--- Processing Finished ---")

# %%
