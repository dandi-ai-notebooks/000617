# This script loads dF/F traces for 5 cells from the first 3000 timepoints and plots them.
# The goal is to verify trace quality, amplitude, and the presence of neural activity.
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Parameters
N_time = 3000
N_cells = 5
url = "https://api.dandiarchive.org/api/assets/d793b12a-4155-4d22-bd3b-3c49672a5f6a/download/"

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# dF/F traces
dfa = nwb.processing["ophys"].data_interfaces["dff"].roi_response_series["traces"]
dff_data = dfa.data[:N_time, :N_cells]
timestamps = dfa.timestamps[:N_time]

plt.figure(figsize=(10, 5))
for k in range(N_cells):
    plt.plot(timestamps, dff_data[:, k], label=f'Cell {k}')
plt.xlabel("Time (s)")
plt.ylabel("dF/F")
plt.title("dF/F Traces for 5 Cells (first 3000 timepoints)")
plt.legend()
plt.tight_layout()
plt.savefig("explore/dff_traces.png")