# This script plots the distribution of start times for the first 200 "movie_clip_A_presentations".
# The output is a histogram saved as explore/stimulus_timings_hist.png.

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

movie_A_intv = nwb.intervals["movie_clip_A_presentations"].to_dataframe()
start_times = movie_A_intv["start_time"].values[:200]  # first 200 presentations

plt.figure(figsize=(8,4))
plt.hist(start_times, bins=30, color="blue", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Presentation count")
plt.title("Histogram of Start Times for First 200\nMovie Clip A Presentations")
plt.tight_layout()
plt.savefig("explore/stimulus_timings_hist.png", dpi=150)
plt.close()