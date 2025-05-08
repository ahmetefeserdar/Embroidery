# run_multiple_patches2.py
# Based on the original run_multiple_patches.py provided.
# Modifications: Wrapped main logic in generate_streamlines_for_level function.

#%%
import sys, os
from pathlib import Path
import logging # Use logging
import json
import pickle
import traceback

# --- Add parent dir to path if needed for common/embroidery ---
# Adjust if your structure is different
# script_dir = Path(__file__).parent
# parent_dir = script_dir.parent
# if str(parent_dir) not in sys.path:
#     sys.path.append(str(parent_dir))

# --- Attempt to import dependencies ---
try:
    import numpy as np
    import numba
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import patches # Keep patches import? Seems unused later.
    from tqdm import tqdm
    from skimage import restoration, img_as_float, color, measure, transform, filters

    # Embroidery specific imports (adjust paths if necessary)
    from embroidery.utils import math as emb_math, summary
    from embroidery.utils.path import chessboard, subsample
    from embroidery.utils.io import write_bundle, EmbPattern
    from embroidery.utils.stitch import stitch_over_path, add_threads
    from embroidery.pipeline import main_pipeline
    from embroidery import gamut

    # Common specific imports (adjust paths if necessary)
    # Assuming common.parse_example exists and is importable
    import common.parse_example
    import common.input # Keep? Seems unused later.
    from common import helpers # Keep? Seems unused later.

except ImportError as e:
    logging.error(f"Failed to import necessary libraries for run_multiple_patches2: {e}")
    logging.error("Please ensure numpy, numba, matplotlib, skimage, tqdm, and the embroidery/common libraries are installed and accessible.")
    # Re-raise or exit if critical dependencies are missing
    raise e


# Configure logging if not already configured by the caller
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Constants (can be passed as args if needed) ---
# DEFAULT_TARGET_TOTAL_PHYSICAL_SIZE_MM = 99 # Renamed to avoid clash
# DEFAULT_PHYSICAL_LINE_WIDTH_MM = 0.4

# --- Numba Functions (defined globally as before) ---
@numba.njit
def recover_colors(grid, color_left, color_right):
    cimg = np.empty((*grid.shape, 3))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            t = grid[i, j]
            cimg[i, j] = (1 - t) * color_left + t * color_right
    return cimg

# Note: These scaling functions now need bbox_min and longest_edge passed explicitly
@numba.njit
def _image_space_to_normal_space(xk, bbox_min_local, longest_edge_local):
    # Add safety check for longest_edge_local being zero
    if longest_edge_local == 0:
        return xk * 0.0 # Or handle as appropriate, e.g., return zeros
    return (xk - bbox_min_local) / longest_edge_local

@numba.njit
def _normal_space_to_image_space(xk, bbox_min_local, longest_edge_local):
    return (xk * longest_edge_local) + bbox_min_local


# --- Main Function Wrapper ---
def generate_streamlines_for_level(
    image_path: Path,
    json_path: Path,
    output_base_dir: Path, # Directory for pickles and level outputs
    target_physical_size_mm: float, # Pass explicitly
    physical_line_width_mm: float, # Pass explicitly
    grid_nx: int = 301, # Grid resolution options
    grid_ny: int = 300,
    subsample_threshold: int = 13, # Stitch subsampling threshold
    save_plots: bool = True # Control saving level plots
):
    """
    Generates foreground and background streamlines for polygons in a given LabelMe JSON file.

    Args:
        image_path: Path to the original image file.
        json_path: Path to the LabelMe JSON file for the current level.
        output_base_dir: Base directory for level-specific outputs (plots, pickles).
        target_physical_size_mm: Target physical size for scaling.
        physical_line_width_mm: Physical line width for relative calculation.
        grid_nx, grid_ny: Resolution for internal grids.
        subsample_threshold: Threshold for subsampling stitches.
        save_plots: Whether to save level preview plots.

    Returns:
        tuple: (EmbPattern object for this level, dict of thread definitions {hex: name})
               Returns (None, {}) on failure.
    """
    task_name = image_path.stem
    level_str = json_path.stem.split('_')[-1] # e.g., L0, L1
    level_output_dir = output_base_dir / f"level_{level_str}_plots" # Specific subdir for plots
    if save_plots:
        level_output_dir.mkdir(exist_ok=True, parents=True) # Create plot dir only if saving

    logging.info(f"Processing level: {level_str} from {json_path}")
    logging.info(f"Level outputs (pickles) in: {output_base_dir}")
    if save_plots: logging.info(f"Level plots in: {level_output_dir}")

    try:
        image = img_as_float(plt.imread(image_path))[:, :, :3]
        # Optional Gaussian blur (already done in segmentation2?)
        # image = filters.gaussian(image, sigma=1)
    except Exception as e:
        logging.error(f"Failed to read image {image_path}: {e}")
        return None, {}

    # --- Load JSON ---
    try:
        with open(json_path) as fp:
            label_json = json.load(fp)
        if not label_json.get("shapes"):
            logging.warning(f"No shapes found in {json_path}. Returning empty pattern for level.")
            return EmbPattern(), {} # Return empty pattern if no shapes
    except Exception as e:
        logging.error(f"Failed to load or parse JSON {json_path}: {e}")
        return None, {}

    # --- Calculate BBox and Scaling ---
    relative_line_width = physical_line_width_mm / target_physical_size_mm
    polygon_shapes = [shape for shape in label_json["shapes"] if shape["shape_type"] == "polygon"]
    if not polygon_shapes:
        logging.warning(f"No polygon shapes found in {json_path}. Returning empty pattern for level.")
        return EmbPattern(), {} # Return empty pattern if no polygons

    # Calculate bounding box based *only* on polygons for consistent scaling
    annotations = np.vstack([np.array(shape["points"]) for shape in polygon_shapes])
    bbox_min_level, bbox_max_level = annotations.min(axis=0), annotations.max(axis=0)
    bbox_dims = bbox_max_level - bbox_min_level
    longest_edge_px_level = max(bbox_dims[0], bbox_dims[1])
    # Handle cases where polygon might be a point or line (0 width/height)
    if longest_edge_px_level <= 0:
        logging.warning(f"Longest edge of bounding box is zero for level {level_str}. Using 1px to avoid division by zero.")
        longest_edge_px_level = 1.0 # Avoid division by zero

    logging.info(f"Level {level_str}: BBox Min={bbox_min_level}, BBox Max={bbox_max_level}, LongestEdge={longest_edge_px_level:.2f}px")
    logging.info(f"Relative line width: {relative_line_width:.5f}")


    # --- Load/Generate Context Map (Pickle) ---
    pickle_fn = output_base_dir / f"{task_name}_{level_str}.pickle"
    logging.info(f"Using pickle file: {pickle_fn}")
    context_map = None
    if pickle_fn.exists():
        try:
            with open(pickle_fn, "rb") as fp:
                context_map = pickle.load(fp)
            logging.info(f"Loaded context map from {pickle_fn}")
        except Exception as e:
            logging.warning(f"Failed to load pickle {pickle_fn}, will regenerate: {e}")
            context_map = None # Ensure regeneration

    if context_map is None:
        # Ensure common.parse_example is available
        if not hasattr(common, 'parse_example') or common.parse_example is None:
             logging.error("common.parse_example module not available. Cannot parse example to generate context map.")
             return None, {}
        # Ensure the necessary function exists
        if not hasattr(common.parse_example, 'parse_example_by_name'):
             logging.error("Function 'parse_example_by_name' not found in common.parse_example.")
             return None, {}

        try:
            logging.info("Generating context map using common.parse_example.parse_example_by_name...")
            # Need to pass scaling info if the parser uses it
            context_map = common.parse_example.parse_example_by_name(
                f"{task_name}_{level_str}", # Use a level-specific identifier
                image,
                label_json,
                # Explicitly pass scaling info if required by parser implementation
                # bbox_min=bbox_min_level,
                # bbox_max=bbox_max_level,
                # longest_edge_px=longest_edge_px_level
            )
            if context_map: # Save only if successfully generated
                with open(pickle_fn, "wb") as fp:
                    pickle.dump(context_map, fp)
                logging.info(f"Saved context map to {pickle_fn}")
            else:
                 logging.warning("Context map generation resulted in empty map.")

        except Exception as e:
            logging.error(f"Failed to generate or save context map: {e}")
            logging.error(traceback.format_exc())
            return None, {}

    if not context_map:
         logging.warning(f"Context map is empty for {level_str}. Returning empty pattern.")
         return EmbPattern(), {} # Return empty pattern if no context


    # --- Generate Streamlines per Region ---
    # Define grid based on passed arguments or defaults
    xl, yl = 0, 0
    xr, yr = 1, 1
    xaxis = np.linspace(xl, xr, grid_nx)
    yaxis = np.linspace(yl, yr, grid_ny)
    xgrid, ygrid = np.meshgrid(xaxis, yaxis)

    logging.info(f"Generating streamlines for {len(context_map)} regions in level {level_str}...")
    # Store generated lines per context item before color projection
    processed_regions = {}

    for i, (label_name, label_ctx) in enumerate(context_map.items()):
        logging.info(f"Processing region {label_name} in {level_str}")
        # Ensure required attributes exist in context
        required_attrs = ['indicator_grid', 'density_grid', 'boundary', 'holes', 'direction_grid', 'color_bg', 'color_fg']
        if not all(hasattr(label_ctx, attr) for attr in required_attrs):
            logging.warning(f"Skipping region {label_name} due to missing attributes in context object.")
            continue

        inside_indicator_grid = label_ctx.indicator_grid
        patch_density_grid = label_ctx.density_grid
        boundary_norm = label_ctx.boundary # Assume these are normalized [0,1]
        holes_norm = label_ctx.holes       # Assume these are normalized [0,1]
        direction_grid = label_ctx.direction_grid

        # Ensure boundary is closed if necessary
        if boundary_norm.shape[0] > 0 and not np.allclose(boundary_norm[0], boundary_norm[-1]):
             boundary_norm = np.vstack((boundary_norm, boundary_norm[0]))

        # Define output folder for this specific region's plots
        region_plot_folder = level_output_dir / f"{label_name}"
        # main_pipeline will create plot folders if plot_save_folder is not None

        try:
            fgl_norm = main_pipeline(
                xaxis=xaxis, yaxis=yaxis,
                boundary=boundary_norm, holes=holes_norm,
                density_grid=patch_density_grid, direction_grid=direction_grid,
                inside_indicator_grid=inside_indicator_grid,
                relative_line_width=relative_line_width,
                plot_save_folder=region_plot_folder / "fg" if save_plots else None,
            )

            rotated_background_direction_grid = direction_grid + np.pi / 2

            bgl_norm = main_pipeline(
                xaxis=xaxis, yaxis=yaxis,
                boundary=boundary_norm, holes=holes_norm,
                density_grid=np.ones_like(xgrid), # Use constant density for background
                direction_grid=rotated_background_direction_grid,
                inside_indicator_grid=inside_indicator_grid,
                relative_line_width=relative_line_width,
                plot_save_folder=region_plot_folder / "bg" if save_plots else None,
            )

            # Store results associated with the context
            processed_regions[label_name] = {
                'ctx': label_ctx,
                'line_fg_norm': fgl_norm,
                'line_bg_norm': bgl_norm
            }

        except Exception as e:
            logging.error(f"Error during main_pipeline for region {label_name} in {level_str}: {e}")
            logging.error(traceback.format_exc())
            # Store empty results for this region to avoid errors later
            processed_regions[label_name] = {
                'ctx': label_ctx, # Keep context for color info
                'line_fg_norm': np.empty((0,2)),
                'line_bg_norm': np.empty((0,2))
            }

    logging.info(f"Pattern generation done for level {level_str}!")

    # --- Color Projection and Pattern Creation for this Level ---
    level_pattern = EmbPattern()
    level_threads = {} # Collect hex -> name mapping for this level {hex: name}

    logging.info(f"Picking colors and building pattern for level {level_str}...")
    # Store image-space lines for potential preview plot
    all_bg_lines_im = []
    all_fg_lines_im = []

    # Use the processed_regions dictionary now
    for label_name, region_data in tqdm(sorted(processed_regions.items(), key=lambda p: p[0])):
        ctx = region_data['ctx']
        bgl_norm = region_data['line_bg_norm']
        fgl_norm = region_data['line_fg_norm']

        if bgl_norm.size == 0 and fgl_norm.size == 0:
             logging.debug(f"Skipping region {label_name} as both BG and FG lines are empty.")
             continue

        # Get original colors and project them
        color_bg_orig = np.clip(ctx.color_bg, 0, 1)
        color_fg_orig = np.clip(ctx.color_fg, 0, 1)

        try:
             # Use default gamut
             default_gamuts = gamut.get_default_gamuts()
             if not default_gamuts:
                 logging.error("Failed to get default gamuts for color projection.")
                 continue # Skip color projection for this region

             (bg_dist, bg_name, bg_hex, _), (fg_dist, fg_name, fg_hex, _) = gamut.gamut_projection_minimum_luminance_distance(
                 [matplotlib.colors.to_hex(color_bg_orig), matplotlib.colors.to_hex(color_fg_orig)],
                 default_gamuts,
             )
             projected_color_bg_hex = f"#{bg_hex}"
             projected_color_fg_hex = f"#{fg_hex}"
             colorname_bg = bg_name[0] # Assuming single name result
             colorname_fg = fg_name[0] # Assuming single name result

             # Store thread info if new
             if projected_color_bg_hex not in level_threads: level_threads[projected_color_bg_hex] = colorname_bg
             if projected_color_fg_hex not in level_threads: level_threads[projected_color_fg_hex] = colorname_fg

             # --- Convert normalized coords to physical coords and add stitches ---
             # Physical coordinates are typically in tenths of millimeters for formats like DST
             scale_factor = target_physical_size_mm * 10

             # Apply subsampling
             phy_bgl = subsample(bgl_norm * scale_factor, subsample_threshold)
             phy_fgl = subsample(fgl_norm * scale_factor, subsample_threshold)

             # Add stitches for this region to the level pattern, including color info
             # Ensure stitch_over_path handles potentially empty arrays
             if phy_bgl.size > 0:
                 stitch_over_path(level_pattern, phy_bgl, color_hex=projected_color_bg_hex, color_breaks=False)
             if phy_fgl.size > 0:
                 stitch_over_path(level_pattern, phy_fgl, color_hex=projected_color_fg_hex, color_breaks=False)

             # --- Store image-space lines for preview plot (if saving plots) ---
             if save_plots:
                 # Convert normalized lines back to image space using level's bbox
                 im_bgl = _normal_space_to_image_space(bgl_norm, bbox_min_level, longest_edge_px_level)
                 im_fgl = _normal_space_to_image_space(fgl_norm, bbox_min_level, longest_edge_px_level)
                 if im_bgl.size > 0: all_bg_lines_im.append((im_bgl, projected_color_bg_hex))
                 if im_fgl.size > 0: all_fg_lines_im.append((im_fgl, projected_color_fg_hex))

        except Exception as e:
             logging.error(f"Error during color projection or stitch generation for region {label_name}: {e}")
             logging.error(traceback.format_exc())


    # --- Generate Preview Plot for the Level (if saving plots) ---
    if save_plots:
        logging.info(f"Generating preview plot for level {level_str}...")
        preview_fig, preview_ax = plt.subplots(figsize=(8, 8))
        preview_ax.imshow(image) # Show original image background

        # Determine plot line width based on image resolution and physical size
        # This is an approximation for visualization
        img_h, img_w = image.shape[:2]
        pixels_per_mm = max(img_h, img_w) / target_physical_size_mm # Approx px/mm
        plot_line_width_px = physical_line_width_mm * pixels_per_mm
        plot_line_width = max(0.5, plot_line_width_px / 10) # Heuristic adjustment for plot line width

        # Plot stitches using image coordinates
        for lines, color in all_bg_lines_im:
             preview_ax.plot(lines[:, 0], lines[:, 1], color=color, linewidth=plot_line_width, solid_capstyle='round')
        for lines, color in all_fg_lines_im:
             preview_ax.plot(lines[:, 0], lines[:, 1], color=color, linewidth=plot_line_width, solid_capstyle='round')

        # Set limits based on image dimensions
        preview_ax.set_xlim(0, img_w)
        preview_ax.set_ylim(img_h, 0) # Y axis inverted for images
        preview_ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
        preview_ax.axis("off")
        preview_plot_path = level_output_dir / f"preview_{level_str}.png"
        try:
            preview_fig.savefig(preview_plot_path, bbox_inches="tight", pad_inches=0, dpi=150)
            logging.info(f"Saved level preview plot to {preview_plot_path}")
        except Exception as e:
            logging.error(f"Failed to save preview plot {preview_plot_path}: {e}")
        plt.close(preview_fig) # Close the figure

    plt.close('all') # Close any figures potentially opened by main_pipeline

    logging.info(f"Finished processing level {level_str}.")
    # Return the generated pattern object and the threads used in this level
    return level_pattern, level_threads

# Note: No `if __name__ == "__main__":` block needed here