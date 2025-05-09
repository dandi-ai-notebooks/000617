# This script generates a heatmap of all cell masks (ROIs) from the imaging session by overlaying all mask images.
# Objective: To illustrate the spatial location/distribution of segmented cells on the imaging plane.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/d793b12a-4155-4d22-bd3b-3c49672a5f6a/download/"

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

roi_table = nwb.processing["ophys"].data_interfaces["dff"].roi_response_series["traces"].rois.table
# roi_table.image_mask is a VectorData of shape (n_cells, 512, 512), pixel values 0-1
masks = np.array([roi_table.image_mask[i] for i in range(len(roi_table.id))])
# Heatmap: display the max projection across all ROI masks (so pixels in many overlap ROIs are bright)
mask_overlay = np.max(masks, axis=0)

plt.figure(figsize=(6, 6))
plt.imshow(mask_overlay, cmap='hot')
plt.title("Overlay of All Cell Masks (ROIs)")
plt.axis('off')
plt.tight_layout()
plt.savefig("explore/cell_mask_overlay.png")