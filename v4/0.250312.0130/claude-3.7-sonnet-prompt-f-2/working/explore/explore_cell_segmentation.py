"""
Explore the cell segmentation masks from the ophys data.
This script aims to:
1. Load the segmentation masks for the cells
2. Visualize them on the average/max projection images
3. Examine spatial organization of cells
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the image segmentation data 
print("Loading segmentation data...")
ophys_module = nwb.processing['ophys']
image_seg = ophys_module.data_interfaces['image_segmentation']
plane_seg = image_seg.plane_segmentations['cell_specimen_table']

# Get cell ROI information
cells = plane_seg.to_dataframe()
print(f"Number of cells: {len(cells)}")

# Print column information
print("Columns:", cells.columns)
print("\nExample cell data:")
print(cells.iloc[0])

# Get spatial information about ROIs
x_pos = cells['x'].values
y_pos = cells['y'].values
width = cells['width'].values
height = cells['height'].values
valid_roi = cells['valid_roi'].values

print(f"\nValid ROIs: {np.sum(valid_roi)}/{len(valid_roi)}")

# Get the max and average projection images
images = ophys_module.data_interfaces['images']
max_proj = np.array(images.images['max_projection'])
avg_proj = np.array(images.images['average_image'])
seg_mask = np.array(images.images['segmentation_mask_image'])

print(f"Max projection shape: {max_proj.shape}")
print(f"Average projection shape: {avg_proj.shape}")
print(f"Segmentation mask shape: {seg_mask.shape}")

# Visualize the max projection image with cell ROIs
plt.figure(figsize=(10, 10))
plt.imshow(max_proj, cmap='gray')
plt.title("Max Projection with Cell ROIs")
plt.scatter(x_pos + width/2, y_pos + height/2, s=20, c='r', alpha=0.7)

for i, (x, y) in enumerate(zip(x_pos, y_pos)):
    plt.text(x+width[i]/2, y+height[i]/2, f"{i}", color='y', fontsize=8)

plt.colorbar(label='Fluorescence intensity')
plt.tight_layout()
plt.savefig('explore/max_projection_with_rois.png')

# Visualize the average projection image with cell ROIs
plt.figure(figsize=(10, 10))
plt.imshow(avg_proj, cmap='gray')
plt.title("Average Projection with Cell ROIs")
plt.scatter(x_pos + width/2, y_pos + height/2, s=20, c='r', alpha=0.7)
plt.colorbar(label='Fluorescence intensity')
plt.tight_layout()
plt.savefig('explore/avg_projection_with_rois.png')

# Visualize the segmentation mask image
plt.figure(figsize=(10, 10))
plt.imshow(seg_mask, cmap='viridis')
plt.title("Segmentation Mask Image")
plt.colorbar(label='ROI ID')
plt.tight_layout()
plt.savefig('explore/segmentation_mask.png')

# Load ROI masks for the first 20 cells
num_cells_to_display = min(20, len(cells))
masks = []
for i in range(num_cells_to_display):
    mask = plane_seg['image_mask'].data[i]
    mask_reshaped = mask.reshape(plane_seg['image_mask'].data[i].shape[0], -1)
    masks.append(mask_reshaped)
    
# Create a composite mask visualization
composite_mask = np.zeros_like(max_proj, dtype=float)
colors = plt.cm.rainbow(np.linspace(0, 1, num_cells_to_display))

for i, mask in enumerate(masks):
    if mask.shape == composite_mask.shape:
        # Add this ROI to the composite with a unique color
        color_mask = np.zeros((*composite_mask.shape, 4))  # RGBA
        for c in range(3):  # RGB channels
            color_mask[..., c] = mask * colors[i, c]
        color_mask[..., 3] = mask * 0.5  # Alpha channel
        
        # Only update non-zero pixels
        idx = mask > 0
        composite_mask[idx] = composite_mask[idx] + 1

# Visualize the composite mask
plt.figure(figsize=(12, 10))
plt.imshow(max_proj, cmap='gray', alpha=0.7)
plt.imshow(composite_mask, cmap='hot', alpha=0.5)
plt.colorbar(label="Number of overlapping ROIs")
plt.title(f"Composite Cell ROI Masks (First {num_cells_to_display} cells)")
plt.tight_layout()
plt.savefig('explore/composite_roi_masks.png')

# Get cell mask image data
print("\nAccessing cell masks...")
cell_masks = []
for i in range(min(5, len(cells))):
    mask_data = plane_seg['image_mask'][i]
    print(f"Cell {i} mask shape: {mask_data.shape}")
    print(f"Cell {i} mask data type: {mask_data.dtype}")
    print(f"Cell {i} mask range: {np.min(mask_data)} to {np.max(mask_data)}")
    imax = np.unravel_index(np.argmax(mask_data), mask_data.shape)
    print(f"Cell {i} max value position: {imax}")

# Try to visualize individual cell masks
plt.figure(figsize=(15, 10))
for i in range(min(6, len(cells))):
    plt.subplot(2, 3, i+1)
    mask_data = plane_seg['image_mask'][i]
    if len(mask_data.shape) == 1:
        # Need to reshape based on available information
        width = int(np.sqrt(len(mask_data)))
        mask_data = mask_data.reshape(width, -1)
    
    plt.imshow(mask_data, cmap='viridis')
    plt.title(f"Cell {i} Mask")
    plt.colorbar()

plt.tight_layout()
plt.savefig('explore/individual_roi_masks.png')