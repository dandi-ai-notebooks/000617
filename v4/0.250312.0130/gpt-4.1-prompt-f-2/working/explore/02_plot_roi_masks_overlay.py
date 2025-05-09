# This script overlays all ROI masks onto the max projection image for a visual summary of segmentation.
# The output plot is saved as explore/roi_masks_overlay.png.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, "r")
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

roi_table = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces'].rois.table
roi_masks = np.array(roi_table.image_mask)  # (121, 512, 512), boolean

# Load the max projection image from Images interface
img_module = nwb.processing['ophys'].data_interfaces["images"]
max_proj = img_module.images["max_projection"].data[:]  # should be 512x512
if max_proj.shape != (512, 512):
    max_proj = max_proj.T  # ensure correct orientation if needed

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(max_proj, cmap="gray")
# Overlay: show a heatmap of mask coverage
mask_heatmap = np.max(roi_masks, axis=0)  # 1 where any ROI present
ax.imshow(mask_heatmap, cmap="hot", alpha=0.3)
ax.set_title("Overlay of All ROI Masks on Max Projection Image")
ax.axis("off")
plt.tight_layout()
plt.savefig("explore/roi_masks_overlay.png", dpi=175)
plt.close()