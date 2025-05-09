# This script explores and plots DFF traces for a few ROIs from the NWB file.

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

# Access DFF traces and timestamps
dff_traces = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces']
timestamps = dff_traces.timestamps[:]

# Select the first 5 ROIs
num_rois_to_plot = 5
roi_indices = np.arange(num_rois_to_plot)
# Load data for the selected ROIs
dff_data = dff_traces.data[:, roi_indices]

# Plot the DFF traces
plt.figure(figsize=(12, 6))
for i in range(num_rois_to_plot):
    plt.plot(timestamps, dff_data[:, i], label=f'ROI {roi_indices[i]}')

plt.xlabel('Time (s)')
plt.ylabel('dF/F')
plt.title('Selected ROI dF/F Traces')
plt.legend()
plt.grid(True)
plt.savefig('explore/dff_traces.png')
plt.close()

io.close()