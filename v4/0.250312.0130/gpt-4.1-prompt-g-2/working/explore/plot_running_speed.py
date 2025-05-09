# This script visualizes the running speed (cm/s) for the session from the NWB file.
# Goal: Assess the presence of running bouts and overall behavior data quality.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

N_time = 3000
url = "https://api.dandiarchive.org/api/assets/d793b12a-4155-4d22-bd3b-3c49672a5f6a/download/"

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

speed_ts = nwb.processing["running"].data_interfaces["speed"]
speed = speed_ts.data[:N_time]
timestamps = speed_ts.timestamps[:N_time]

plt.figure(figsize=(10, 4))
plt.plot(timestamps, speed, color='tab:orange')
plt.xlabel("Time (s)")
plt.ylabel("Running speed (cm/s)")
plt.title("Running Speed (first 3000 timepoints)")
plt.tight_layout()
plt.savefig("explore/running_speed.png")