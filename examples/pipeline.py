#%%
# pipeline.py
import sys
import os
import argparse
import logging
from pathlib import Path
import json
import pickle
from collections import OrderedDict
import re # For sorting level files
import traceback # For detailed error logging
#%%
# --- Configure Logging ---
# Set level to DEBUG for more verbose output if needed
log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

#%%
# --- Add project root to sys.path --- START ADDITION
try:
    # Get the directory containing this script (pipeline.py, expected to be in 'examples/')
    script_dir = Path(__file__).resolve().parent
    # Get the parent directory (expected to be the project root containing 'embroidery/' and 'common/')
    project_root = script_dir.parent

    # Add the project root directory to the start of the Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO: Added project root to sys.path: {project_root}")
        # You can uncomment the next line to help debug if needed:
        # print(f"INFO: Contents of added directory: {os.listdir(project_root)}")
except NameError:
     # Fallback if __file__ is not defined (e.g., interactive execution)
     print("WARNING: Could not automatically determine project root. Assuming it's already in sys.path or the current directory.")
# --- END ADDITION ---


#%%
# --- Import necessary components from the *copied and modified* scripts ---
try:
    logging.info("Importing from segmentation2...")
    from segmentation2 import run_segmentation
    logging.info("Importing from run_multiple_patches2...")
    from run_multiple_patches2 import generate_streamlines_for_level

    # Import utilities needed for final output
    # These might be within run_multiple_patches2 or directly from embroidery
    logging.info("Importing from embroidery utils...")
    from embroidery.utils.io import write_bundle, EmbPattern
    from embroidery.utils.stitch import add_threads
    from embroidery.utils import summary

    # Ensure matplotlib backend is suitable for non-interactive use if needed
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend to avoid GUI displays during saving
    logging.info("Matplotlib backend set to Agg.")

except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Please ensure segmentation2.py, run_multiple_patches2.py, and dependencies (numpy, skimage, matplotlib, embroidery, common, etc.) are installed and accessible in the Python path.")
    logging.error(traceback.format_exc())
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during imports: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)

#%%
# --- Constants (Consider making these argparse arguments too) ---
DEFAULT_TARGET_PHYSICAL_SIZE_MM = 99
DEFAULT_PHYSICAL_LINE_WIDTH_MM = 0.4
DEFAULT_GRID_NX = 301
DEFAULT_GRID_NY = 300
DEFAULT_SUBSAMPLE_THRESHOLD = 13

#%%
def sort_level_files(file_list: list[Path]) -> list[Path]:
    """Sorts labelme JSON files based on the L<number> suffix."""
    def get_level(path):
        match = re.search(r'_L(\d+)\.json$', path.name)
        return int(match.group(1)) if match else float('inf')
    return sorted(file_list, key=get_level)

def main():
    parser = argparse.ArgumentParser(description="Run the full segmentation and streamline generation pipeline using copied scripts.")
    parser.add_argument("--image", required=True, type=str, help="Path to the input image file (e.g., sky2.jpg).")
    parser.add_argument("--threshold", required=True, type=float, help="Segmentation merge threshold (e.g., 2.8).")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="Base directory for all pipeline outputs.")
    parser.add_argument("--skip_segmentation", action="store_true", help="Skip segmentation step and use existing JSON files.")
    parser.add_argument("--save_plots", action="store_true", default=False, help="Enable saving of intermediate plots (segmentation, level previews).")
    # Add more arguments for finer control if needed
    parser.add_argument("--physical_size", type=float, default=DEFAULT_TARGET_PHYSICAL_SIZE_MM, help="Target physical size in mm.")
    parser.add_argument("--line_width", type=float, default=DEFAULT_PHYSICAL_LINE_WIDTH_MM, help="Physical line width in mm.")
    parser.add_argument("--grid_nx", type=int, default=DEFAULT_GRID_NX, help="Grid resolution Nx for streamline generation.")
    parser.add_argument("--grid_ny", type=int, default=DEFAULT_GRID_NY, help="Grid resolution Ny for streamline generation.")
    parser.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE_THRESHOLD, help="Subsampling threshold for stitches.")
    # Include segmentation params if you want to override defaults from segmentation2.py
    parser.add_argument("--n_segments", type=int, help="Override default N_SEGMENTS for SLIC.")
    # ... add more segmentation parameter arguments as needed ...

    args = parser.parse_args()

    input_image_path = Path(args.image).resolve() # Use resolved absolute path
    if not input_image_path.is_file():
        logging.error(f"Input image not found: {input_image_path}")
        sys.exit(1)

    segmentation_threshold = args.threshold
    base_output_dir = Path(args.output_dir).resolve()
    image_stem = input_image_path.stem
    segmentation_output_dir = base_output_dir / image_stem / "segmentation_output"
    streamline_output_dir = base_output_dir / image_stem / "streamline_output" # Pickles go here
    final_output_dir = base_output_dir / image_stem / "final_pattern"

    # Create output directories
    segmentation_output_dir.mkdir(parents=True, exist_ok=True)
    streamline_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting pipeline for image: {input_image_path}")
    logging.info(f"Segmentation Threshold: {segmentation_threshold}")
    logging.info(f"Base Output Directory: {base_output_dir}")
    logging.info(f"Save Intermediate Plots: {args.save_plots}")

    # === Step 1: Run Segmentation (using segmentation2.run_segmentation) ===
    json_files = []
    if not args.skip_segmentation:
        logging.info("--- Running Segmentation (using segmentation2.py) ---")
        try:
            # Prepare segmentation arguments, allowing overrides from command line
            seg_args = {
                "image_path_str": str(input_image_path),
                "threshold": segmentation_threshold,
                "output_dir": segmentation_output_dir,
                "save_plots": args.save_plots
            }
            if args.n_segments: seg_args["n_segments"] = args.n_segments
            # Add other overridden params here...

            img_path_obj, H, W, generated_files = run_segmentation(**seg_args)

            if img_path_obj is None or not generated_files: # Check if segmentation function indicated failure
                logging.error("Segmentation function failed or produced no JSON files. Stopping.")
                sys.exit(1)
            json_files = generated_files # These are Path objects
            logging.info(f"Segmentation successful. Found {len(json_files)} JSON level file(s).")
        except Exception as e:
            logging.error(f"Segmentation script (segmentation2.py) failed with an error: {e}")
            logging.error(traceback.format_exc())
            sys.exit(1)
    else:
        logging.info("--- Skipping Segmentation ---")
        # Find existing JSON files in the segmentation output directory
        json_files = list(segmentation_output_dir.glob(f"{image_stem}_labelme_L*.json"))
        if not json_files:
            logging.error(f"Skip segmentation specified, but no JSON files found in {segmentation_output_dir}")
            sys.exit(1)
        logging.info(f"Found {len(json_files)} existing JSON level file(s).")

    # Sort the JSON files by level number
    sorted_json_files = sort_level_files(json_files)
    if not sorted_json_files:
         logging.error("No valid JSON level files found or sorted. Stopping.")
         sys.exit(1)

    logging.info("Processing JSON files in order:")
    for f in sorted_json_files: logging.info(f"  - {f.name}")

    # === Step 2: Run Streamline Generation (using run_multiple_patches2.generate_streamlines_for_level) ===
    level_patterns = OrderedDict() # Store patterns keyed by level number
    all_threads = {} # Collect all unique threads across levels {hex: name}
    streamline_generation_failed = False # Flag for overall success

    logging.info("--- Generating Streamlines for Each Level (using run_multiple_patches2.py) ---")
    for json_path in sorted_json_files:
        level_match = re.search(r'_L(\d+)\.json$', json_path.name)
        if not level_match:
            logging.warning(f"Could not extract level number from {json_path.name}. Skipping.")
            continue
        level_num = int(level_match.group(1))

        try:
            logging.info(f"--- Processing Level {level_num} ---")
            level_pattern, level_threads = generate_streamlines_for_level(
                input_image_path,
                json_path,
                streamline_output_dir, # Pickles and level-specific plots go here/subdirs
                target_physical_size_mm=args.physical_size,
                physical_line_width_mm=args.line_width,
                grid_nx=args.grid_nx,
                grid_ny=args.grid_ny,
                subsample_threshold=args.subsample,
                save_plots=args.save_plots
            )

            if level_pattern is not None:
                level_patterns[level_num] = level_pattern
                # Merge threads safely
                for hex_code, name in level_threads.items():
                    if hex_code in all_threads and all_threads[hex_code] != name:
                        logging.warning(f"Thread color {hex_code} has conflicting names: '{all_threads[hex_code]}' vs '{name}'. Keeping first encountered ('{all_threads[hex_code]}').")
                    elif hex_code not in all_threads:
                        all_threads[hex_code] = name
                logging.info(f"Successfully generated pattern for Level {level_num} ({len(level_threads)} threads in level, {len(level_pattern.stitches if hasattr(level_pattern, 'stitches') else [])} stitches).")
            else:
                # Function returned None, indicating failure for this level
                logging.error(f"Streamline generation FAILED for Level {level_num}.")
                streamline_generation_failed = True # Mark failure

        except Exception as e:
            logging.error(f"Streamline generation script (run_multiple_patches2.py) for Level {level_num} ({json_path.name}) failed with an error: {e}")
            logging.error(traceback.format_exc())
            streamline_generation_failed = True # Mark failure
            logging.warning("Attempting to continue to the next level...")

    if not level_patterns:
         logging.error("No level patterns were successfully generated. Cannot create final output.")
         sys.exit(1)
    if streamline_generation_failed:
         logging.warning("One or more levels failed during streamline generation. The final pattern might be incomplete.")
         # Decide whether to proceed or exit? Let's proceed but warn.


    # === Step 3: Merge Level Patterns ===
    logging.info("--- Merging Level Patterns ---")
    combined_pattern = EmbPattern()

    # Add all collected threads (even from failed levels if they were partially processed)
    if all_threads:
        add_threads(combined_pattern, list(all_threads.keys()), list(all_threads.values()))
        logging.info(f"Added {len(combined_pattern.threads)} unique threads to the combined pattern.")
    else:
        logging.warning("No threads collected from any level.")

    # Append stitches sequentially from successfully processed levels (L0, L1, L2...)
    stich_count = 0
    for level_num in sorted(level_patterns.keys()): # Iterate through successfully generated patterns
        logging.info(f"Adding stitches from Level {level_num}")
        level_pat = level_patterns[level_num]
        if hasattr(level_pat, 'stitches') and level_pat.stitches:
             num_level_stitches = len(level_pat.stitches)
             combined_pattern.stitches.extend(level_pat.stitches)
             stich_count += num_level_stitches
             logging.info(f"  Added {num_level_stitches} stitches.")
        else:
             logging.warning(f"  Level {level_num} pattern had no stitches to add.")

    logging.info(f"Total stitches in combined pattern: {stich_count}")
    if stich_count == 0:
         logging.warning("Combined pattern has zero stitches.")


    # === Step 4: Save Final Output ===
    logging.info("--- Saving Final Combined Pattern ---")
    try:
        # Output final summary
        print("\n" + "="*30 + " Final Combined Pattern Summary " + "="*30)
        print(summary.summarize_pattern(combined_pattern))
        print("="*80)

        # Write final embroidery files
        output_basename = f"{image_stem}_final_T{segmentation_threshold:.2f}".replace('.', '_') # Include threshold in name
        write_bundle(combined_pattern, final_output_dir, output_basename)
        logging.info(f"Final embroidery files saved with basename '{output_basename}' in {final_output_dir}")

        # Optionally, save the combined pattern object itself
        combined_pattern_pickle = final_output_dir / f"{output_basename}.embpattern.pickle"
        with open(combined_pattern_pickle, "wb") as f:
            pickle.dump(combined_pattern, f)
        logging.info(f"Saved combined EmbPattern object to {combined_pattern_pickle}")

    except Exception as e:
        logging.error(f"Failed to save the final combined pattern: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

    if streamline_generation_failed:
        logging.warning("--- Pipeline Completed with Errors (some levels may have failed) ---")
    else:
        logging.info("--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()