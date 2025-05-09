# This script explores and visualizes some of the ROI masks from the NWB file.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the cell specimen table and image masks
cell_specimen_table = nwb.processing['ophys'].data_interfaces['image_segmentation'].plane_segmentations['cell_specimen_table']
image_masks = cell_specimen_table.image_mask
mask_image_planes = cell_specimen_table.mask_image_plane[:]

# Select a few ROI indices to visualize
roi_indices_to_plot = [0, 10, 20, 30] # Example indices

plt.figure(figsize=(10, 10))
for i, roi_index in enumerate(roi_indices_to_plot):
    # Get the mask for this ROI
    mask = image_masks[roi_index]
    # Plot the mask
    plt.subplot(2, 2, i + 1)
    plt.imshow(mask, cmap='gray')
    plt.title(f'ROI {roi_index} (Plane {mask_image_planes[roi_index]})')
    plt.axis('off')

plt.tight_layout()
plt.savefig('explore/roi_masks_individual.png')
plt.close()

# Now visualize some superimposed masks from different planes
planes_to_combine = [0, 1] # Example planes
combined_mask = None
for plane in planes_to_combine:
    plane_masks = [image_masks[i] for i, p in enumerate(mask_image_planes) if p == plane]
    if plane_masks:
        stacked_masks = np.stack(plane_masks, axis=0)
        if combined_mask is None:
            combined_mask = np.max(stacked_masks, axis=0)
        else:
            combined_mask = np.max(np.stack([combined_mask, np.max(stacked_masks, axis=0)], axis=0), axis=0)

if combined_mask is not None:
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_mask, cmap='gray')
    plt.title(f'Superimposed ROI Masks from Planes {planes_to_combine}')
    plt.axis('off')
    plt.savefig('explore/roi_masks_superimposed.png')
    plt.close()

io.close()