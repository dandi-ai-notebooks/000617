# explore_script_2.py
# This script loads the NWB file and plots the max_projection image.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get max_projection image data
max_projection_data = nwb.processing['ophys'].data_interfaces['images'].images['max_projection'].data[:]

# Plot the image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(max_projection_data, cmap='gray')
ax.set_title('Max Projection Image')
ax.set_xlabel('X pixels')
ax.set_ylabel('Y pixels')
plt.tight_layout()
plt.savefig('explore/max_projection.png')
plt.close(fig) # Close the figure to prevent hanging

print("Max projection image saved to explore/max_projection.png")

io.close()