# This script plots dF/F traces for a sample of ROIs (first 6) to illustrate population activity over time.
# The output plot is saved as explore/dff_traces_sample.png.

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

dff = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces']
timestamps = dff.timestamps[:]
data = dff.data[:, :6]  # (40019, 6)
roi_ids = list(dff.rois.table.id[:6])

fig, axs = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
for i in range(6):
    axs[i].plot(timestamps, data[:, i], lw=0.4)
    axs[i].set_ylabel(f'ROI {roi_ids[i]}')
axs[0].set_title('Example dF/F Calcium Traces for 6 ROIs')
axs[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig("explore/dff_traces_sample.png", dpi=150)
plt.close()