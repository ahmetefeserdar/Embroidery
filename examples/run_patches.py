# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#%%
import sys
import os
from pathlib import Path
import json
import pickle
from typing import List, Tuple, Dict, Any # Added for type hinting
import logging

# Ensure the parent directory is in the path if common and embroidery are there
try:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
        logging.debug(f"Added {parent_dir} to sys.path")
except NameError:
    script_dir = Path.cwd()
    logging.debug(f"Running interactively or __file__ not defined. Using current working directory: {script_dir}")
    parent_dir = script_dir.parent
    if parent_dir.is_dir() and str(parent_dir) not in sys.path:
       sys.path.append(str(parent_dir))
       logging.debug(f"Added {parent_dir} to sys.path based on cwd.")


import matplotlib
from matplotlib import pyplot as plt

from tqdm import tqdm

import numpy as np
import numba

from skimage import io, img_as_float, filters, color

try:
    from shapely.geometry import Polygon, LineString, MultiPolygon
except ImportError:
    print("CRITICAL ERROR: Shapely library not found. Please install it (`pip install shapely`)")
    sys.exit(1)

# Embroidery and Common module imports
try:
    from embroidery.utils import math as embroidery_math, summary as embroidery_summary
    from embroidery.pipeline import main_pipeline
    from embroidery import gamut
    from embroidery.utils.path import chessboard, subsample
    from embroidery.utils.io import write_bundle, EmbPattern # Keep EmbPattern if needed by parser
    from embroidery.utils.stitch import stitch_over_path, add_threads # Keep if needed by parser/context
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'embroidery' modules: {e}. Ensure library is installed/in path.")
    sys.exit(1)

try:
    import common.parse_example
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'common' modules: {e}. Ensure directory is structured correctly/in path.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# %%
# --- Script Configuration ---
task_name_base = "bird2"  # << SET YOUR BASE TASK NAME HERE
source_data_folder = Path("data")
output_base_folder = Path("output")
SKIP_LEVEL_0 = True # Set to True to skip background/Level 0 embroidery

# Create output folders
merged_output_folder = output_base_folder / f"{task_name_base}_merged"
intermediate_plots_dir = merged_output_folder / "intermediate_plots"
os.makedirs(merged_output_folder, exist_ok=True)
os.makedirs(intermediate_plots_dir, exist_ok=True)
logging.info(f"Output will be saved to: {merged_output_folder.resolve()}")
if SKIP_LEVEL_0: logging.info("Configuration set to SKIP Level 0 processing.")

# Embroidery parameters (still needed for main_pipeline)
target_total_physical_size_mm = 99.0
physical_line_width_mm = 0.4
if target_total_physical_size_mm <= 0:
     logging.warning("target_total_physical_size_mm is zero or negative. Using default relative_line_width.")
     relative_line_width = 0.004
else:
     relative_line_width = physical_line_width_mm / target_total_physical_size_mm

# --- Load Base Image & Get Dimensions ---
base_image_fn = source_data_folder / f"{task_name_base}.jpg"
base_image_display = None
base_image_for_parsing = None
IMG_H, IMG_W = 0, 0 # Will be determined from image or first JSON

if base_image_fn.exists():
    try:
        img_raw = io.imread(base_image_fn)
        if img_raw.ndim == 2: img_raw = color.gray2rgb(img_raw)
        if img_raw.shape[2] == 4: img_raw = img_raw[:,:,:3]
        IMG_H, IMG_W = img_raw.shape[:2]
        logging.info(f"Loaded base image: {base_image_fn.resolve()} (Dimensions: {IMG_W}x{IMG_H})")
        base_image_for_parsing = filters.gaussian(img_as_float(img_raw), sigma=1)
        base_image_display = img_as_float(img_raw)
    except Exception as e:
        logging.error(f"Could not load/process base image {base_image_fn}: {e}")
        # Try to get dimensions from JSON later if image load fails but we need them
else:
    logging.warning(f"Base image {base_image_fn} not found.")
    # Try to get dimensions from the first JSON file found
    first_json_path = next(source_data_folder.glob(f"{task_name_base}_labelme_L*.json"), None)
    if first_json_path:
        try:
            with open(first_json_path) as fp: first_json_data = json.load(fp)
            IMG_H = first_json_data.get('imageHeight'); IMG_W = first_json_data.get('imageWidth')
            if IMG_H and IMG_W: logging.info(f"Using dimensions from {first_json_path.name}: {IMG_W}x{IMG_H}")
            else: raise ValueError("Dimensions not found in JSON")
        except Exception as e: logging.error(f"Could not read dimensions from {first_json_path.name}: {e}. Exiting."); sys.exit(1)
    else: logging.error(f"No base image or level JSON files found to infer dimensions. Exiting."); sys.exit(1)

# Check if dimensions were successfully obtained
if IMG_W <= 0 or IMG_H <= 0:
     logging.error(f"Invalid image dimensions obtained ({IMG_W}x{IMG_H}). Exiting.")
     sys.exit(1)

# --- Find level JSON files ---
level_json_files = sorted(list(source_data_folder.glob(f"{task_name_base}_labelme_L*.json")))
if not level_json_files: logging.error(f"No level JSON files found for '{task_name_base}'."); sys.exit(1)
logging.info(f"Found {len(level_json_files)} level JSON files: {[f.name for f in level_json_files]}")

# --- Initialize Master Accumulators ---
# Removed accumulators related to final embroidery pattern
master_preview_lines_fg: List[Tuple[np.ndarray, str]] = []
master_preview_lines_bg: List[Tuple[np.ndarray, str]] = []


# %%
# --- Loop Through Each Level JSON and Process ---
for level_idx, json_fn_level in enumerate(level_json_files):
    
    is_level_0 = json_fn_level.stem.endswith("_L0")
    if SKIP_LEVEL_0 and is_level_0:
        logging.info(f"--- Skipping Level 0 JSON: {json_fn_level.name} ---")
        continue

    logging.info(f"--- Processing Level JSON: {json_fn_level.name} ---")

    with open(json_fn_level) as fp: label_json_current_level = json.load(fp)

    # --- Calculate LOCAL Bounding Box for this level ---
    current_level_polygon_annotations = []
    for shape in label_json_current_level.get("shapes", []):
        if shape.get("shape_type") == "polygon" and shape.get("points"):
            try:
                points_np = np.array(shape["points"], dtype=float)
                if points_np.ndim == 2 and points_np.shape[0] >= 3 and points_np.shape[1] == 2 and np.all(np.isfinite(points_np)):
                     current_level_polygon_annotations.append(points_np)
            except ValueError: pass

    if not current_level_polygon_annotations:
        logging.warning(f"No valid polygon annotations found in {json_fn_level.name}. Skipping this file.")
        continue
    
    all_points_for_bbox = np.vstack(current_level_polygon_annotations)
    bbox_min_local, bbox_max_local = all_points_for_bbox.min(axis=0), all_points_for_bbox.max(axis=0)
    width = bbox_max_local[0] - bbox_min_local[0]; height = bbox_max_local[1] - bbox_min_local[1]
    if width < 0 or height < 0: logging.error(f"    Invalid bbox for {json_fn_level.name}. Skipping."); continue
    longest_edge_px_local = max(width, height, 1e-6) # Use max with epsilon
    logging.debug(f"    Local Bbox for {json_fn_level.name}: min={bbox_min_local}, max={bbox_max_local}, longest_edge={longest_edge_px_local:.2f}px")

    # --- Define LOCAL transformation functions & Assign to Parser Context ---
    _bbox_min_l = bbox_min_local.copy(); _longest_edge_l = longest_edge_px_local
    @numba.njit
    def _image_space_to_normal_space_local_jit(xk_local_img_coords: np.ndarray) -> np.ndarray:
        xk_float = xk_local_img_coords.astype(np.float64); min_float = _bbox_min_l.astype(np.float64); edge_float = np.float64(_longest_edge_l)
        if edge_float == 0: return xk_float - min_float
        return (xk_float - min_float) / edge_float
    @numba.njit
    def _normal_space_to_image_space_local_jit(xk_norm_coords: np.ndarray) -> np.ndarray:
        xk_float = xk_norm_coords.astype(np.float64); min_float = _bbox_min_l.astype(np.float64); edge_float = np.float64(_longest_edge_l)
        return (xk_float * edge_float) + min_float

    if hasattr(common, 'parse_example'):
        # Assign LOCAL helpers before parsing this level
        common.parse_example._image_space_to_normal_space = _image_space_to_normal_space_local_jit
        common.parse_example._normal_space_to_image_space = _normal_space_to_image_space_local_jit
        common.parse_example.bbox_min = _bbox_min_l
        common.parse_example.longest_edge_px = _longest_edge_l
        logging.debug(f"    Assigned LOCAL normalization helpers to common.parse_example for {json_fn_level.name}")
    else: logging.error("common.parse_example module not found. Cannot assign helpers."); continue

    # --- Context Map Generation ---
    try: parsed_level_num = int(json_fn_level.stem.split('_L')[-1])
    except (IndexError, ValueError): parsed_level_num = level_idx; logging.warning(f"Could not parse level num from {json_fn_level.name}.")
    level_specific_cache_name = f"{task_name_base}_L{parsed_level_num}"
    pickle_fn_level = source_data_folder / f"{level_specific_cache_name}.pickle"
    context_map_current_level: Dict[str, Any] = {}

    if pickle_fn_level.exists():
        try:
            with open(pickle_fn_level, "rb") as fp: context_map_current_level = pickle.load(fp)
            logging.info(f"    Loaded context map from cache: {pickle_fn_level.name}")
        except Exception as e: logging.warning(f"    Could not load context map cache {pickle_fn_level.name}: {e}. Reparsing."); context_map_current_level = {}

    if not context_map_current_level:
        logging.info(f"    Parsing context for {json_fn_level.name}...")
        image_arg_for_parser = base_image_for_parsing if base_image_for_parsing is not None else np.zeros((10,10,3), dtype=np.float32)
        # Parser is now expected to use the LOCAL helpers set above and return NORMALIZED boundary/holes
        context_map_current_level = common.parse_example.parse_example_by_name(level_specific_cache_name, image_arg_for_parser, label_json_current_level)
        try:
            with open(pickle_fn_level, "wb") as fp: pickle.dump(context_map_current_level, fp)
            logging.info(f"    Saved context map to cache: {pickle_fn_level.name}")
        except Exception as e: logging.error(f"    Could not save context map cache {pickle_fn_level.name}: {e}")
    
    if not context_map_current_level: logging.error(f"    Failed to load/parse context_map for {json_fn_level.name}. Skipping."); continue

    # Define grids for main_pipeline (normalized 0-1 space)
    xl_norm, yl_norm=0.0, 0.0; xr_norm, yr_norm=1.0, 1.0
    nx_grid_density, ny_grid_density = 501, 500
    xaxis_norm = np.linspace(xl_norm, xr_norm, nx_grid_density)
    yaxis_norm = np.linspace(yl_norm, yr_norm, ny_grid_density)

    # --- Process each region ---
    for region_idx_in_level, (label_name, label_ctx) in enumerate(context_map_current_level.items()):
        logging.info(f"      Processing region: '{label_name}' (from {json_fn_level.name})")

        # --- VALIDATION of Parser Output (Assumed ALREADY Normalized Locally) ---
        boundary_for_pipeline = None; holes_for_pipeline = []
        try:
            # 1. Check boundary from parser
            if not hasattr(label_ctx, 'boundary') or not isinstance(label_ctx.boundary, np.ndarray): logging.warning(f"        Skipping region '{label_name}': 'boundary' missing/invalid."); continue
            if not np.all(np.isfinite(label_ctx.boundary)): logging.warning(f"        Skipping region '{label_name}': Parsed 'boundary' non-finite."); continue
            if not (label_ctx.boundary.ndim == 2 and label_ctx.boundary.shape[0] >= 3 and label_ctx.boundary.shape[1] == 2): logging.warning(f"        Skipping region '{label_name}': Parsed 'boundary' invalid shape {label_ctx.boundary.shape}."); continue
            
            # 2. Validate with Shapely (using the assumed normalized coords)
            #    No need for explicit normalization here anymore.
            try:
                coords_list = [tuple(pt) for pt in label_ctx.boundary]; assert len(coords_list) >= 3
                normalized_polygon = Polygon(coords_list).buffer(0)
                if not normalized_polygon.is_valid or normalized_polygon.area < 1e-9: logging.warning(f"        Skipping region '{label_name}': Parsed boundary invalid/negligible area."); continue
                # Use exterior coords from validated polygon
                boundary_for_pipeline = np.array(normalized_polygon.exterior.coords)
            except Exception as shapely_error: logging.error(f"        Skipping region '{label_name}': Shapely validation error: {shapely_error}"); continue

            # 3. Validate holes (assuming normalized from parser)
            holes_input = getattr(label_ctx, 'holes', [])
            holes_for_pipeline = []
            if holes_input:
                for i, h in enumerate(holes_input):
                     if isinstance(h, np.ndarray) and h.ndim == 2 and h.shape[0] >= 3 and h.shape[1] == 2 and np.all(np.isfinite(h)):
                          try:
                              hole_coords_list = [tuple(pt) for pt in h]; assert len(hole_coords_list) >= 3
                              hole_poly = Polygon(hole_coords_list).buffer(0)
                              if hole_poly.is_valid and hole_poly.area > 1e-9: holes_for_pipeline.append(np.array(hole_poly.exterior.coords))
                          except Exception: pass # Ignore invalid holes

            # 4. Final check on boundary shape
            if boundary_for_pipeline is None or not (boundary_for_pipeline.ndim == 2 and boundary_for_pipeline.shape[0] >= 3 and boundary_for_pipeline.shape[1] == 2):
                 logging.error(f"        Skipping region '{label_name}': Final boundary shape invalid."); continue

        except Exception as valid_error:
            logging.error(f"        Error during validation for region '{label_name}': {valid_error}. Skipping."); continue
        # --- END VALIDATION ---

        # Validate grids
        density_grid_input = getattr(label_ctx, 'density_grid', None); direction_grid_input = getattr(label_ctx, 'direction_grid', None); inside_indicator_grid_input = getattr(label_ctx, 'indicator_grid', None)
        grid_shape_expected = (ny_grid_density, nx_grid_density)
        if density_grid_input is None or not isinstance(density_grid_input, np.ndarray) or density_grid_input.shape != grid_shape_expected: density_grid_input = np.ones(grid_shape_expected)
        if direction_grid_input is None or not isinstance(direction_grid_input, np.ndarray) or direction_grid_input.shape != grid_shape_expected: direction_grid_input = np.zeros(grid_shape_expected)
        if inside_indicator_grid_input is None or not isinstance(inside_indicator_grid_input, np.ndarray) or inside_indicator_grid_input.shape != grid_shape_expected: inside_indicator_grid_input = np.ones(grid_shape_expected, dtype=bool)

        # Create intermediate plot folder
        level_region_plot_folder = intermediate_plots_dir / json_fn_level.stem / label_name
        os.makedirs(level_region_plot_folder, exist_ok=True)

        # --- Call main_pipeline ---
        fgl_norm, bgl_norm = None, None
        try:
            fgl_norm = main_pipeline(xaxis=xaxis_norm, yaxis=yaxis_norm, boundary=boundary_for_pipeline, holes=holes_for_pipeline, density_grid=density_grid_input, direction_grid=direction_grid_input, inside_indicator_grid=inside_indicator_grid_input, relative_line_width=relative_line_width, plot_save_folder=str(level_region_plot_folder / "fg"))
        except ValueError as ve:
             if 'Cannot apply_along_axis' in str(ve): logging.error(f"        Caught known ValueError in main_pipeline (FG) for region '{label_name}': {ve}.")
             else: logging.error(f"        ValueError in main_pipeline (FG) for region '{label_name}': {ve}", exc_info=False)
        except Exception as pipeline_error: logging.error(f"        Unexpected Error in main_pipeline (FG) for region '{label_name}': {pipeline_error}", exc_info=False)

        try:
            density_grid_bg = np.ones(grid_shape_expected); rotated_background_direction_grid = direction_grid_input + (np.pi / 2)
            bgl_norm = main_pipeline(xaxis=xaxis_norm, yaxis=yaxis_norm, boundary=boundary_for_pipeline, holes=holes_for_pipeline, density_grid=density_grid_bg, direction_grid=rotated_background_direction_grid, inside_indicator_grid=inside_indicator_grid_input, relative_line_width=relative_line_width, plot_save_folder=str(level_region_plot_folder / "bg"))
        except ValueError as ve:
             if 'Cannot apply_along_axis' in str(ve): logging.error(f"        Caught known ValueError in main_pipeline (BG) for region '{label_name}': {ve}.")
             else: logging.error(f"        ValueError in main_pipeline (BG) for region '{label_name}': {ve}", exc_info=False)
             bgl_norm = None
        except Exception as pipeline_error: logging.error(f"        Unexpected Error in main_pipeline (BG) for region '{label_name}': {pipeline_error}", exc_info=False); bgl_norm = None

        # --- Accumulate results ---
        color_bg_rgb = np.clip(getattr(label_ctx, 'color_bg', [0.1,0.1,0.1]), 0, 1)
        color_fg_rgb = np.clip(getattr(label_ctx, 'color_fg', [0.9,0.9,0.9]), 0, 1)
        try: bg_hex = matplotlib.colors.to_hex(color_bg_rgb); fg_hex = matplotlib.colors.to_hex(color_fg_rgb)
        except ValueError: bg_hex, fg_hex = "#101010", "#E0E0E0"
        bg_color_name = f"bg_{label_name}_L{parsed_level_num}"; fg_color_name = f"fg_{label_name}_L{parsed_level_num}"

        # Accumulate preview paths (Convert back to IMAGE SPACE using LOCAL transform)
        if bgl_norm is not None and bgl_norm.size > 0: master_preview_lines_bg.append((_normal_space_to_image_space_local_jit(bgl_norm), bg_hex))
        if fgl_norm is not None and fgl_norm.size > 0: master_preview_lines_fg.append((_normal_space_to_image_space_local_jit(fgl_norm), fg_hex))

logging.info(f"--- All processed level JSONs finished. Preview data collected. ---")

# %%
# --- Generate Final Combined Preview (After the Loop) ---
logging.info("Generating final combined preview...")
fig_preview, ax_preview = plt.subplots(figsize=(10, 10))
fig_preview.patch.set_facecolor('white'); ax_preview.set_facecolor('white')

# Define fixed line widths for preview (adjust as needed)
preview_line_width_bg = 0.6
preview_line_width_fg = 0.8
logging.debug(f"Using fixed preview line widths: BG={preview_line_width_bg}, FG={preview_line_width_fg}")

# Plot collected lines (Clipping is important here)
for path_coords, hex_color in master_preview_lines_bg:
    if isinstance(path_coords, np.ndarray) and path_coords.ndim == 2 and path_coords.shape[0] > 1 and path_coords.shape[1] == 2 :
        clipped_path = path_coords.copy(); clipped_path[:, 0] = np.clip(clipped_path[:, 0], 0, IMG_W -1); clipped_path[:, 1] = np.clip(clipped_path[:, 1], 0, IMG_H -1)
        ax_preview.plot(clipped_path[:, 0], clipped_path[:, 1], color=hex_color, linewidth=preview_line_width_bg) # Use fixed width
for path_coords, hex_color in master_preview_lines_fg:
    if isinstance(path_coords, np.ndarray) and path_coords.ndim == 2 and path_coords.shape[0] > 1 and path_coords.shape[1] == 2 :
        clipped_path = path_coords.copy(); clipped_path[:, 0] = np.clip(clipped_path[:, 0], 0, IMG_W -1); clipped_path[:, 1] = np.clip(clipped_path[:, 1], 0, IMG_H -1)
        ax_preview.plot(clipped_path[:, 0], clipped_path[:, 1], color=hex_color, linewidth=preview_line_width_fg) # Use fixed width

# Finalize preview plot
if IMG_W > 0 and IMG_H > 0: ax_preview.set_xlim(0, IMG_W); ax_preview.set_ylim(IMG_H, 0)
ax_preview.set_aspect('equal', adjustable='box'); ax_preview.axis("off")
ax_preview.set_title(f"Combined Embroidery Preview - {task_name_base}")
try:
    preview_png_path = merged_output_folder / f"preview_merged.png"; preview_svg_path = merged_output_folder / f"preview_merged.svg"
    plt.savefig(preview_png_path, bbox_inches="tight", pad_inches=0.05, dpi=300, facecolor='white')
    plt.savefig(preview_svg_path, bbox_inches="tight", pad_inches=0.05, facecolor='white')
    logging.info(f"Saved combined preview to {preview_png_path.resolve()} and .svg")
except Exception as e: logging.error(f"Failed to save combined preview: {e}")

plt.show() # Ensure plot is displayed
plt.close(fig_preview) # Close figure after display/save

# %%
# --- Stitching Section Removed as requested ---
logging.info(f"--- Preview generation for '{task_name_base}' finished. Stitching section skipped. ---")
