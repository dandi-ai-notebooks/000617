"""
Explore the ΔF/F traces from the ophys data.
This script aims to:
1. Load and visualize ΔF/F data for a subset of neurons
2. Examine responses to different movie stimuli
3. Create visualizations showing neuron activity patterns
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb
import pandas as pd
import seaborn as sns

# Set up Seaborn
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the ΔF/F data
print("Loading ΔF/F data...")
dff = nwb.processing["ophys"]["dff"]
dff_traces = dff.roi_response_series["traces"]

# Get timestamps and data for a small subset
timestamps = dff_traces.timestamps[:]
print(f"Total timepoints: {len(timestamps)}")
print(f"Recording duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"Sampling rate: {1/np.mean(np.diff(timestamps)):.2f} Hz")

# Get a subset of neurons (first 10)
n_neurons = 10
dff_data = dff_traces.data[:, :n_neurons]
print(f"DFF data shape: {dff_data.shape}")

# Get cell metadata
cell_table = dff_traces.rois.table
cell_ids = cell_table.cell_specimen_id[:n_neurons]
cell_locations = list(zip(
    cell_table.x[:n_neurons], 
    cell_table.y[:n_neurons]
))
print(f"Cell IDs: {', '.join(str(id) for id in cell_ids)}")
print(f"Cell locations: {cell_locations}")

# Get stimulus information
stim_gray = nwb.intervals["gray_presentations"]
stim_A = nwb.intervals["movie_clip_A_presentations"] 
stim_B = nwb.intervals["movie_clip_B_presentations"]
stim_C = nwb.intervals["movie_clip_C_presentations"]

# Get all stimuli as a dataframe for easier analysis
print("Creating stimulus dataframe...")
gray_df = stim_gray.to_dataframe().iloc[:100]  # Just get first 100 to save time
A_df = stim_A.to_dataframe().iloc[:100]
B_df = stim_B.to_dataframe().iloc[:100]
C_df = stim_C.to_dataframe().iloc[:100]

print(f"Gray stim: {gray_df.shape[0]} presentations")
print(f"Movie A: {A_df.shape[0]} presentations")
print(f"Movie B: {B_df.shape[0]} presentations")
print(f"Movie C: {C_df.shape[0]} presentations")

# Function to find data indices within a time range
def get_indices_in_range(timestamps, start_time, end_time):
    return np.where((timestamps >= start_time) & (timestamps < end_time))[0]

# Plot ΔF/F traces for a single neuron over time
plt.figure(figsize=(14, 6))

# Choose one neuron to visualize
neuron_idx = 0
trace = dff_data[:, neuron_idx]
plt.plot(timestamps, trace)
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F')
plt.title(f'ΔF/F Trace for Cell ID {cell_ids[neuron_idx]}')

# Add some stimulus windows to see if we can detect responses
# Just add first few of each stimulus type
for i, (name, df) in enumerate([
    ('Gray', gray_df.iloc[:3]), 
    ('Movie A', A_df.iloc[:3]), 
    ('Movie B', B_df.iloc[:3]),
    ('Movie C', C_df.iloc[:3])
]):
    for idx, row in df.iterrows():
        start, end = row['start_time'], row['stop_time']
        if start > timestamps[0] and end < timestamps[-1]:
            plt.axvspan(start, end, alpha=0.2, color=f'C{i+1}')

# Save the figure
plt.tight_layout()
plt.savefig('explore/dff_single_neuron_trace.png')

# Plot activity heatmap of multiple neurons
plt.figure(figsize=(14, 8))
time_window = 200  # seconds - limit plot to first 200 seconds
time_indices = timestamps < timestamps[0] + time_window
plt.imshow(
    dff_data[time_indices, :].T,
    aspect='auto',
    interpolation='none',
    extent=[timestamps[0], timestamps[0] + time_window, 0, n_neurons],
    cmap='viridis'
)
plt.colorbar(label='ΔF/F')
plt.ylabel('Neuron #')
plt.xlabel('Time (s)')
plt.title('ΔF/F Activity Heatmap for Multiple Neurons')

# Add stimulus annotations
stimulus_colors = {
    'Gray': 'gray',
    'Movie A': 'red',
    'Movie B': 'blue', 
    'Movie C': 'green'
}

for name, df in [
    ('Gray', gray_df), 
    ('Movie A', A_df), 
    ('Movie B', B_df),
    ('Movie C', C_df)
]:
    for idx, row in df.iterrows():
        start, end = row['start_time'], row['stop_time']
        if start > timestamps[0] and end < timestamps[0] + time_window:
            plt.axvline(x=start, color=stimulus_colors[name], alpha=0.5, linewidth=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=name) 
                   for name, color in stimulus_colors.items()]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('explore/dff_heatmap.png')

# Compute average responses to each stimulus type for a single neuron
neuron_idx = 0
window_size = 20  # 20 timepoints before and after stimulus

# Get responses to each stimulus type
responses = {}
max_frames = 30  # Limit to first 30 frames after stimulus onset for visualization
for name, df in [
    ('Gray', gray_df.iloc[:10]),  # First 10 of each
    ('Movie A', A_df.iloc[:10]), 
    ('Movie B', B_df.iloc[:10]),
    ('Movie C', C_df.iloc[:10])
]:
    all_responses = []
    for idx, row in df.iterrows():
        start = row['start_time']
        # Find frames around stimulus onset
        onset_idx = np.argmin(np.abs(timestamps - start))
        if onset_idx + max_frames < len(timestamps):
            response = dff_data[onset_idx:onset_idx + max_frames, neuron_idx]
            all_responses.append(response)
    
    # Only store if we have responses
    if all_responses:
        responses[name] = np.array(all_responses)

# Plot average responses
plt.figure(figsize=(12, 6))
for name, resp_array in responses.items():
    if len(resp_array) > 0:
        mean_response = np.mean(resp_array, axis=0)
        sem_response = np.std(resp_array, axis=0) / np.sqrt(resp_array.shape[0])
        
        time_vec = np.arange(len(mean_response)) / 10.0  # Convert to seconds (10Hz sampling)
        plt.plot(time_vec, mean_response, label=name)
        plt.fill_between(
            time_vec,
            mean_response - sem_response,
            mean_response + sem_response,
            alpha=0.2
        )

plt.axvline(x=0, linestyle='--', color='gray')
plt.xlabel('Time from stimulus onset (s)')
plt.ylabel('ΔF/F')
plt.title(f'Average Responses to Different Stimuli for Cell ID {cell_ids[neuron_idx]}')
plt.legend()
plt.tight_layout()
plt.savefig('explore/average_responses.png')

# Plot ROI locations
plt.figure(figsize=(8, 8))
x_coords = cell_table.x[:n_neurons]
y_coords = cell_table.y[:n_neurons]

plt.scatter(x_coords, y_coords, s=30)
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    plt.text(x+2, y+2, f"{cell_ids[i]}", fontsize=8)

plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.title('ROI Locations')
plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
plt.tight_layout()
plt.savefig('explore/roi_locations.png')