# explore_script_5.py
# This script loads the NWB file and plots the animal's running speed.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get running speed data
running_speed_ts = nwb.processing['running'].data_interfaces['speed']
speed_data = running_speed_ts.data[:] # Load all for potential slicing later
speed_timestamps = running_speed_ts.timestamps[:]

# Select a subset of data to plot
num_samples_to_plot = 2000
speed_subset = speed_data[:num_samples_to_plot]
timestamps_subset = speed_timestamps[:num_samples_to_plot]

# Plot the running speed
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(timestamps_subset, speed_subset)
ax.set_title(f'Running Speed (First {num_samples_to_plot} samples)')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Speed ({running_speed_ts.unit})')
plt.tight_layout()
plt.savefig('explore/running_speed.png')
plt.close(fig)

print(f"Running speed plot saved to explore/running_speed.png. Plotted {len(timestamps_subset)} samples.")
actual_duration_plotted = timestamps_subset[-1] - timestamps_subset[0] if len(timestamps_subset) > 1 else 0
print(f"Actual duration plotted: {actual_duration_plotted:.2f} seconds.")


io.close()