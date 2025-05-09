# explore_script_3.py
# This script loads the NWB file and plots all ROI image masks superimposed.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get cell specimen table and image masks
cell_table = nwb.processing['ophys']['image_segmentation'].plane_segmentations['cell_specimen_table']
image_masks_series = cell_table['image_mask']
# Determine the shape of the imaging plane from the first mask
# It's typically (height, width) as stored, but imshow expects (width, height) traditionally if not specified
# However, image_mask from AllenSDK is usually (y,x) or (height,width)
# Let's assume the shape of the first mask is representative for creating the composite.
first_mask_shape = image_masks_series[0].shape
composite_mask = np.zeros(first_mask_shape, dtype=float)

for i in range(len(image_masks_series)):
    mask = image_masks_series[i]
    # Ensure mask is boolean or can be safely converted to float where True is 1.0
    composite_mask = np.maximum(composite_mask, mask.astype(float))

# Plot the composite mask
fig, ax = plt.subplots(figsize=(8, 8))
# The image_mask values are 0 or 1 (or boolean). Using a binary colormap might be too stark.
# 'viridis' or 'gray' can show intensity if masks overlap or have weights (though here they are binary).
# Using np.max ensures we just see the union of masks.
im = ax.imshow(composite_mask, cmap='viridis', interpolation='nearest') # Using viridis for better visibility of overlaps if any
ax.set_title('Superimposed ROI Image Masks')
ax.set_xlabel('X pixels')
ax.set_ylabel('Y pixels')
plt.colorbar(im, ax=ax, label='Max mask value')
plt.tight_layout()
plt.savefig('explore/roi_masks.png')
plt.close(fig)

print("ROI masks image saved to explore/roi_masks.png")

io.close()