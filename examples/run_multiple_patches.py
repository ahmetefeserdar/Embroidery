# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#%%
import sys, os; sys.path.append("..")

import common.parse_example; 

import json
import pickle
import examples.common.parse_example


import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches

from tqdm import tqdm

import numpy as np
import numba

from skimage import restoration, img_as_float, color, measure, transform, filters

from embroidery.utils import math, summary
from embroidery.pipeline import main_pipeline
from embroidery import gamut

import common.input
from common import helpers

from pathlib import Path
from IPython import get_ipython
if (ipython_instance := get_ipython()) is not None:
	ipython_instance.run_line_magic("load_ext", "autoreload")
	ipython_instance.run_line_magic("autoreload", "2")

# %%
task_name = f"sky2"
image_fn = f"data/{task_name}.jpg"
json_fn = f"data/{task_name}_kmeans_L0.json"


image = img_as_float(plt.imread(image_fn))[:, :, :3]
image = filters.gaussian(image, sigma=1)
plt.imshow(image)

# make output dir, otherwise boom
os.makedirs(f"output/{task_name}", exist_ok=True)

#%%
target_total_physical_size_mm = 99#124
physical_line_width_mm = 0.4

@numba.njit
def recover_colors(grid, color_left, color_right):
	cimg = np.empty((*grid.shape, 3))
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			t = grid[i, j]
			cimg[i, j] = (1 - t) * color_left + t * color_right
	return cimg


#%%
with open(json_fn) as fp:
	label_json = json.load(fp)
	print([s["label"] for s in label_json["shapes"]])

target_physical_size_mm = 99
relative_line_width = physical_line_width_mm / target_physical_size_mm

annotations = np.vstack(
	[np.array(shape["points"]) for shape in label_json["shapes"] if shape["shape_type"] == "polygon"]
	)
bbox_min, bbox_max = annotations.min(axis=0), annotations.max(axis=0)
longest_edge_px = max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1])

print(f"{bbox_min = }, {bbox_max = }")
print(f"{(bbox_max - bbox_min) / longest_edge_px * target_physical_size_mm}mm")

@numba.njit
def _image_space_to_normal_space(xk):
	return (xk - bbox_min) / longest_edge_px

@numba.njit
def _normal_space_to_image_space(xk):
	return (xk * longest_edge_px) + bbox_min

#%%
sys.modules["parse_example"] = common.parse_example
import common.parse_example as example_context # real import
common.example_context = example_context # expose attribute


if Path(f"data/{task_name}.pickle").exists():
	with open(f"data/{task_name}.pickle", "rb") as fp:
		context_map = pickle.load(fp)
else:
	context_map = common.example_context.parse_example_by_name(task_name, image, label_json)
	with open(f"data/{task_name}.pickle", "wb") as fp:
		pickle.dump(context_map, fp)

xl, yl = 0, 0
xr, yr = 1, 1
nx, ny = 501, 500
xaxis = np.linspace(xl, xr, nx)
yaxis = np.linspace(yl, yr, ny)
xgrid, ygrid = np.meshgrid(xaxis, yaxis)


#%%
keys = sorted(context_map.keys())
fg_lines = []
bg_lines = []
for i, (label_name, label_ctx) in enumerate(context_map.items()):
	print(f"processing {label_name}")
	inside_indicator_grid = label_ctx.indicator_grid
	patch_density_grid = label_ctx.density_grid

	assert np.allclose(label_ctx.boundary[0], label_ctx.boundary[-1])

	fgl = main_pipeline(
		xaxis=xaxis,
		yaxis=yaxis,
		boundary=label_ctx.boundary,
		holes=label_ctx.holes,
		density_grid=patch_density_grid,
		direction_grid=label_ctx.direction_grid,
		inside_indicator_grid=inside_indicator_grid,
		relative_line_width=relative_line_width,
		plot_save_folder=f"output/{task_name}/{label_name}-fg",
	)

	rotated_background_direction_grid = label_ctx.direction_grid + np.pi / 2

	bgl = main_pipeline(
		xaxis=xaxis,
		yaxis=yaxis,
		boundary=label_ctx.boundary,
		holes=label_ctx.holes,
		density_grid=np.ones_like(xgrid),
		direction_grid=rotated_background_direction_grid,
		inside_indicator_grid=inside_indicator_grid,
		relative_line_width=relative_line_width,
		plot_save_folder=f"output/{task_name}/{label_name}-bg",
	)

	label_ctx.line_bg = bgl
	label_ctx.line_fg = fgl

print("pattern generation done!")

"""
 Color projection to our palette, output preview 
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xr = xmax - xmin
yr = ymax - ymin

color_names = []
color_hexes = []

# project to the nearest color
print("picking colors...")
for label_name, ctx in tqdm(sorted([(k, v) for k, v in context_map.items()], key=lambda p: p[0])):
	# print(label_name)
	color_bg = np.clip(ctx.color_bg, 0, 1)
	color_fg = np.clip(ctx.color_fg, 0, 1)
	im_bgl = np.apply_along_axis(_normal_space_to_image_space, axis=1, arr=ctx.line_bg)
	im_fgl = np.apply_along_axis(_normal_space_to_image_space, axis=1, arr=ctx.line_fg)


	(bg_dist, bg_name, bg_hex, _), (fg_dist, fg_name, fg_hex, _) = gamut.gamut_projection_minimum_luminance_distance(
		[matplotlib.colors.to_hex(color_bg), matplotlib.colors.to_hex(color_fg)],
		gamut.get_default_gamuts(),
	)


	ctx.colorname_bg = bg_name[0]
	ctx.colorname_fg = fg_name[0]
	ctx.projected_color_bg = f"#{bg_hex}"
	ctx.projected_color_fg = f"#{fg_hex}"

	color_names.append(bg_name)
	color_names.append(fg_name)
	color_hexes.append(ctx.projected_color_bg)
	color_hexes.append(ctx.projected_color_fg)
	print(bg_name, bg_dist)
	print(fg_name, fg_dist)

	bg_lp, = plt.plot(im_bgl[:, 0], im_bgl[:, 1], color=f"#{bg_hex}")
	fg_lp, = plt.plot(im_fgl[:, 0], im_fgl[:, 1], color=f"#{fg_hex}")
	bg_lp.set_linewidth(1 / xr * relative_line_width * 350)
	fg_lp.set_linewidth(1 / xr * relative_line_width * 350)

plt.axis("scaled")
plt.axis("off")
plt.savefig(f"output/{task_name}/preview.png", bbox_inches="tight", pad_inches=0)
plt.savefig(f"output/{task_name}/preview.svg", bbox_inches="tight", pad_inches=0)
plt.show()
"""
#%% Color projection to our palette, output preview
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
# Check if xr and yr can be calculated, otherwise set to a default to avoid division by zero if plot is empty
if xmax > xmin and ymax > ymin:
    xr = xmax - xmin
    yr = ymax - ymin
else:
    xr = 1.0 # Default value
    yr = 1.0 # Default value
    logging.warning("Plot limits were not set prior to color projection; preview line widths might be incorrect.")


color_names = []
color_hexes = []

# project to the nearest color
print("picking colors (projection commented out, using original colors)...") # Modified print
for label_name, ctx in tqdm(sorted([(k, v) for k, v in context_map.items()], key=lambda p: p[0])):
    # print(label_name)
    color_bg = np.clip(ctx.color_bg, 0, 1) # Original background color
    color_fg = np.clip(ctx.color_fg, 0, 1) # Original foreground color
    im_bgl = np.apply_along_axis(_normal_space_to_image_space, axis=1, arr=ctx.line_bg)
    im_fgl = np.apply_along_axis(_normal_space_to_image_space, axis=1, arr=ctx.line_fg)

    # --- Lines to comment out for disabling projection ---
    # (bg_dist, bg_name, bg_hex, _), (fg_dist, fg_name, fg_hex, _) = gamut.gamut_projection_minimum_luminance_distance(
    #     [matplotlib.colors.to_hex(color_bg), matplotlib.colors.to_hex(color_fg)],
    #     gamut.get_default_gamuts(),
    # )
    # --- End of lines to comment out ---

    # --- Use original colors instead of projected ones ---
    # Convert original RGB (0-1 float) to hex for compatibility with existing code
    try:
        bg_hex = matplotlib.colors.to_hex(color_bg, keep_alpha=False)
        fg_hex = matplotlib.colors.to_hex(color_fg, keep_alpha=False)
    except ValueError as e:
        logging.error(f"Error converting color to hex for {label_name}: bg={color_bg}, fg={color_fg}. Error: {e}")
        bg_hex = "#000000" # Fallback to black
        fg_hex = "#FFFFFF" # Fallback to white

    bg_name = [f"original_bg_{label_name}"] # Generic name
    fg_name = [f"original_fg_{label_name}"] # Generic name
    bg_dist = 0 # No projection, so distance is 0
    fg_dist = 0 # No projection, so distance is 0
    # --- End of using original colors ---

    # These attributes on ctx might be used later, so assign them
    ctx.colorname_bg = bg_name[0]
    ctx.colorname_fg = fg_name[0]
    ctx.projected_color_bg = bg_hex # Using original hex
    ctx.projected_color_fg = fg_hex # Using original hex

    color_names.append(bg_name[0]) # Storing the generic name
    color_names.append(fg_name[0]) # Storing the generic name
    color_hexes.append(ctx.projected_color_bg) # Storing original hex
    color_hexes.append(ctx.projected_color_fg) # Storing original hex

    # These print statements will now show the generic names and 0 distance
    print(bg_name[0], bg_dist)
    print(fg_name[0], fg_dist)

    # Plotting will use the original colors (converted to hex)
    # Ensure im_bgl and im_fgl are not empty before plotting
    if im_bgl.size > 0:
        bg_lp, = plt.plot(im_bgl[:, 0], im_bgl[:, 1], color=bg_hex) # Use bg_hex (original)
        if xr > 0: # Avoid division by zero
          bg_lp.set_linewidth(1 / xr * relative_line_width * 350)
    else:
        logging.warning(f"No background lines to plot for {label_name}")

    if im_fgl.size > 0:
        fg_lp, = plt.plot(im_fgl[:, 0], im_fgl[:, 1], color=fg_hex) # Use fg_hex (original)
        if xr > 0: # Avoid division by zero
          fg_lp.set_linewidth(1 / xr * relative_line_width * 350)
    else:
        logging.warning(f"No foreground lines to plot for {label_name}")


plt.axis("scaled")
plt.axis("off")
plt.savefig(f"output/{task_name}/preview.png", bbox_inches="tight", pad_inches=0)
plt.savefig(f"output/{task_name}/preview.svg", bbox_inches="tight", pad_inches=0)
plt.show()

#%%
from embroidery.utils.path import chessboard, subsample
from embroidery.utils.io import write_bundle, EmbPattern
from embroidery.utils.stitch import stitch_over_path, add_threads

pattern = EmbPattern()
add_threads(pattern, color_hexes, color_names)
# stitch_over_path(pattern, chessboard(1, 1, 999, 999, 11), color_breaks=True)

for label_name, ctx in sorted([(k, v) for k, v in context_map.items()], key=lambda p: p[0]):
	fgl, bgl = ctx.line_fg, ctx.line_bg
	phy_fgl = subsample(fgl * target_total_physical_size_mm * 10, 13)
	phy_bgl = subsample(bgl * target_total_physical_size_mm * 10, 13)

	print("bg", summary.summarize_pattern(phy_bgl))
	print("fg", summary.summarize_pattern(phy_fgl))
	stitch_over_path(pattern, phy_bgl, color_breaks=True)
	stitch_over_path(pattern, phy_fgl, color_breaks=True)

print(summary.summarize_pattern(pattern))
write_bundle(pattern, f"output/stitch", f"{task_name}-stitch")
write_bundle(pattern, f"output/{task_name}", f"{task_name}")
# %%
