# This script loads the running speed data and plots it over time.
# The purpose is to visualize the animal's locomotion during the experiment.

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

# Access the running speed data
# Path: processing/running/speed
speed_dataset = nwb.processing['running']['speed'].data
speed_timestamps_dataset = nwb.processing['running']['speed'].timestamps

# Load a subset of data for plotting efficiency
# Let's load the first 50000 time points
num_time_points = 50000

speed_subset = speed_dataset[0:num_time_points]
speed_timestamps_subset = speed_timestamps_dataset[0:num_time_points]

# Plot the running speed
plt.figure(figsize=(12, 4))
plt.plot(speed_timestamps_subset, speed_subset)
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Subset of Running Speed over Time')
plt.tight_layout()

# Save the plot to a file
plt.savefig('explore/running_speed_subset.png')
plt.close() # Close the plot to prevent it from displaying

# Close the NWB file
io.close()