# %% Visualise regions stored in a LabelMe-style JSON file (no file output)
import json, random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

json_path = Path("data/bird2_labelme_L1.json")   # ← change if needed
linewidth  = 1.5                                  # outline thickness

# ---------- read polygons ----------------------------------------------------
with open(json_path) as f:
    data = json.load(f)

patches, colours = [], []
for shape in data["shapes"]:
    patches.append(Polygon(shape["points"], closed=True, edgecolor="none"))
    colours.append([random.random()*0.6 + 0.2 for _ in range(3)])  # pastel

# ---------- draw -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 9))
coll = PatchCollection(patches, facecolor=colours, edgecolor="k",
                       linewidth=linewidth, alpha=0.45)
ax.add_collection(coll)

# optional: label each region
for patch, shape in zip(patches, data["shapes"]):
    x, y = patch.get_xy().mean(axis=0)
    ax.text(x, y, shape["label"], ha="center", va="center", fontsize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground="w")])

ax.set_aspect("equal")
ax.autoscale_view()
ax.axis("off")

plt.show()
# %%
