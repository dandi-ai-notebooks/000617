# This script loads a subset of dF/F traces from the NWB file and plots them.
# The purpose is to visualize neural activity over time for a few selected ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file - this part is hardcoded based on nwb-file-info output
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the dF/F traces
# Path: processing/ophys/dff/roi_response_series/traces
dff_traces_dataset = nwb.processing['ophys']['dff'].roi_response_series['traces'].data
timestamps_dataset = nwb.processing['ophys']['dff'].roi_response_series['traces'].timestamps

# Load a subset of data to avoid excessive memory usage and download time
# Let's load the first 10000 time points and the first 5 ROIs
num_time_points = 10000
num_rois = 5

dff_subset = dff_traces_dataset[0:num_time_points, 0:num_rois]
timestamps_subset = timestamps_dataset[0:num_time_points]

# Load ROI information to get cell specimen IDs
roi_table = nwb.processing['ophys']['dff'].roi_response_series['traces'].rois.table.to_dataframe()
cell_specimen_ids = roi_table['cell_specimen_id'].iloc[0:num_rois].tolist()


# Plot the traces
plt.figure(figsize=(12, 6))
for i in range(num_rois):
    plt.plot(timestamps_subset, dff_subset[:, i] + i * 10, label=f'ROI {cell_specimen_ids[i]}') # Offset traces for clarity

plt.xlabel('Time (s)')
plt.ylabel('dF/F (arbitrary units, offset for clarity)')
plt.title('Subset of dF/F Traces for Selected ROIs')
plt.legend()
plt.tight_layout()

# Save the plot to a file
plt.savefig('explore/dff_traces_subset.png')
plt.close() # Close the plot to prevent it from displaying

# Close the NWB file
io.close()