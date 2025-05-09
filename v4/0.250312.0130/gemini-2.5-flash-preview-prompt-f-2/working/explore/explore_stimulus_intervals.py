# This script loads the stimulus presentation intervals for movie_clip_A and plots them as colored bars.
# The purpose is to visualize when a particular stimulus was presented during the experiment.

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

# Access the stimulus presentation intervals for movie_clip_A
# Path: intervals/movie_clip_A_presentations
stimulus_intervals_table = nwb.intervals['movie_clip_A_presentations'].to_dataframe()

# Get the start and stop times
start_times = stimulus_intervals_table['start_time'].values
stop_times = stimulus_intervals_table['stop_time'].values

# Plot the intervals
plt.figure(figsize=(12, 2))
for start, stop in zip(start_times, stop_times):
    plt.barh(y=[0], width=stop - start, left=start, height=1, color='blue', edgecolor='none')

plt.xlabel('Time (s)')
plt.yticks([]) # Hide y-axis ticks
plt.title('Presentation Intervals for Movie Clip A (Subset)')
plt.xlim([start_times[0], start_times[0] + 60]) # Show only the first minute as an example
plt.tight_layout()

# Save the plot to a file
plt.savefig('explore/stimulus_intervals_subset.png')
plt.close() # Close the plot to prevent it from displaying

# Close the NWB file
io.close()