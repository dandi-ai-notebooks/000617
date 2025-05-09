# This script explores and plots the running speed from the NWB file.

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

# Access running speed and timestamps
running_speed = nwb.processing['running'].data_interfaces['speed']
timestamps = running_speed.timestamps[:]
speed_data = running_speed.data[:]

# Plot the running speed
plt.figure(figsize=(12, 6))
plt.plot(timestamps, speed_data)

plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Running Speed Over Time')
plt.grid(True)
plt.savefig('explore/running_speed.png')
plt.close()

io.close()