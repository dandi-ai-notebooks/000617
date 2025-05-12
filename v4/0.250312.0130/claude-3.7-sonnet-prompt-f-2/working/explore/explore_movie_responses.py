"""
Explore responses to movie stimuli in the ophys data.
This script aims to:
1. Examine how neurons respond to different movie clips
2. Analyze temporal patterns of responses
3. Visualize population-level activity during stimulus presentations
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
timestamps = dff_traces.timestamps[:]
dff_data = dff_traces.data[:, :]  # All neurons

print(f"Total timepoints: {len(timestamps)}")
print(f"Recording duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"Number of neurons: {dff_data.shape[1]}")

# Get stimulus information
stim_gray = nwb.intervals["gray_presentations"]
stim_A = nwb.intervals["movie_clip_A_presentations"] 
stim_B = nwb.intervals["movie_clip_B_presentations"]
stim_C = nwb.intervals["movie_clip_C_presentations"]

# Convert to dataframes for easier analysis
gray_df = stim_gray.to_dataframe()
A_df = stim_A.to_dataframe()
B_df = stim_B.to_dataframe()
C_df = stim_C.to_dataframe()

print(f"Gray stim: {gray_df.shape[0]} presentations")
print(f"Movie A: {A_df.shape[0]} presentations")
print(f"Movie B: {B_df.shape[0]} presentations")
print(f"Movie C: {C_df.shape[0]} presentations")

# Find time indices for each stimulus presentation
def get_indices_in_range(timestamps, start_time, end_time):
    return np.where((timestamps >= start_time) & (timestamps < end_time))[0]

# Function to compute average response for a set of presentations
def compute_average_response(dff_data, timestamps, stim_df, n_cells=None):
    if n_cells is None:
        n_cells = dff_data.shape[1]
    
    # Use just the first 50 presentations to keep computation manageable
    stim_df = stim_df.iloc[:50]
    
    # Find the average response pattern during each presentation
    all_responses = []
    presentation_count = 0
    
    for idx, row in stim_df.iterrows():
        start, end = row['start_time'], row['stop_time']
        time_indices = get_indices_in_range(timestamps, start, end)
        
        if len(time_indices) > 0:
            # Time vector relative to stimulus start
            stim_times = timestamps[time_indices] - start
            
            # Get responses for all cells
            responses = dff_data[time_indices, :n_cells]
            all_responses.append((stim_times, responses))
            presentation_count += 1
    
    print(f"  Processed {presentation_count} presentations")
    return all_responses

# Compute average responses for all stimulus types
print("Computing responses to stimuli...")
gray_responses = compute_average_response(dff_data, timestamps, gray_df, n_cells=30)
A_responses = compute_average_response(dff_data, timestamps, A_df, n_cells=30)
B_responses = compute_average_response(dff_data, timestamps, B_df, n_cells=30)
C_responses = compute_average_response(dff_data, timestamps, C_df, n_cells=30)

# Get a cell ROI information
cell_table = dff_traces.rois.table
cell_ids = cell_table.cell_specimen_id[:]

# Find neurons with strong preference for particular movie types
# To do this, we'll compute the mean response during each movie type
print("Identifying neurons with stimulus preferences...")

def compute_mean_response_by_stimulus(dff_data, timestamps, stim_df, n_cells):
    """Compute the mean response for each cell during a stimulus type"""
    # Use just the first 100 presentations to keep computation manageable
    stim_df = stim_df.iloc[:100]
    
    # Initialize the mean response array
    mean_responses = np.zeros(n_cells)
    total_frames = 0
    
    for idx, row in stim_df.iterrows():
        start, end = row['start_time'], row['stop_time']
        time_indices = get_indices_in_range(timestamps, start, end)
        
        if len(time_indices) > 0:
            # Sum the dF/F values during this stimulus presentation
            mean_responses += np.sum(dff_data[time_indices, :n_cells], axis=0)
            total_frames += len(time_indices)
    
    # Compute the mean
    if total_frames > 0:
        mean_responses /= total_frames
    
    return mean_responses

n_cells = min(100, dff_data.shape[1])  # Analyze up to 100 cells

# Compute mean response for each stimulus type
gray_mean = compute_mean_response_by_stimulus(dff_data, timestamps, gray_df, n_cells)
A_mean = compute_mean_response_by_stimulus(dff_data, timestamps, A_df, n_cells)
B_mean = compute_mean_response_by_stimulus(dff_data, timestamps, B_df, n_cells)
C_mean = compute_mean_response_by_stimulus(dff_data, timestamps, C_df, n_cells)

# Find neurons with strongest preference for each movie type
movie_preferences = np.zeros(n_cells, dtype=int)  # 0=gray, 1=A, 2=B, 3=C
stimulus_names = ['Gray', 'Movie A', 'Movie B', 'Movie C']

for i in range(n_cells):
    responses = [gray_mean[i], A_mean[i], B_mean[i], C_mean[i]]
    movie_preferences[i] = np.argmax(responses)

# Count cells preferring each stimulus
preference_counts = [np.sum(movie_preferences == i) for i in range(4)]
print("Cells preferring each stimulus type:")
for i, count in enumerate(preference_counts):
    print(f"  {stimulus_names[i]}: {count} cells")

# Plot a heatmap of neuron responses to each stimulus type
plt.figure(figsize=(10, 12))
response_matrix = np.column_stack([gray_mean, A_mean, B_mean, C_mean])

# Sort neurons by their preferred stimulus
sort_indices = np.lexsort((np.arange(n_cells), movie_preferences))
sorted_response_matrix = response_matrix[sort_indices, :]

# Create a heatmap
plt.imshow(sorted_response_matrix, aspect='auto', cmap='viridis')
plt.colorbar(label='Mean ΔF/F')
plt.xlabel('Stimulus Type')
plt.ylabel('Neuron #')
plt.title('Neuron Responses to Different Stimuli')
plt.xticks(np.arange(4), stimulus_names)
plt.tight_layout()
plt.savefig('explore/stimulus_preference_heatmap.png')

# Create bar plot of preference distribution
plt.figure(figsize=(8, 6))
plt.bar(stimulus_names, preference_counts, color=['gray', 'tab:orange', 'tab:green', 'tab:red'])
plt.xlabel('Stimulus Type')
plt.ylabel('Number of Cells')
plt.title('Distribution of Stimulus Preferences')
plt.tight_layout()
plt.savefig('explore/stimulus_preference_counts.png')

# Plot example cells with different preferences
plt.figure(figsize=(15, 12))
cell_types = []

# Find example cells for each preference
for pref in range(4):
    cells_with_pref = np.where(movie_preferences == pref)[0]
    if len(cells_with_pref) > 0:
        # Choose the first cell with this preference
        cell_types.append((cells_with_pref[0], pref))

# Plot average traces for these example cells
n_cells_to_plot = len(cell_types)
for i, (cell_idx, pref) in enumerate(cell_types):
    plt.subplot(n_cells_to_plot, 1, i+1)
    
    # Plot average response for each stimulus type
    for j, (name, responses) in enumerate(zip(
        stimulus_names, 
        [gray_responses, A_responses, B_responses, C_responses]
    )):
        # Average all presentations for this cell
        all_times = []
        all_resp = []
        
        for stim_times, resp_data in responses:
            all_times.extend(stim_times)
            all_resp.extend(resp_data[:, cell_idx])
        
        # Sort by time
        sort_idx = np.argsort(all_times)
        times_sorted = np.array(all_times)[sort_idx]
        resp_sorted = np.array(all_resp)[sort_idx]
        
        # Create bins for averaging
        bin_width = 0.05  # 50ms bins
        max_time = min(2.0, np.max(times_sorted))  # Cap at 2 seconds
        bins = np.arange(0, max_time, bin_width)
        binned_resp = np.zeros(len(bins)-1)
        count = np.zeros(len(bins)-1)
        
        for t, r in zip(times_sorted, resp_sorted):
            if t < max_time:
                bin_idx = int(t / bin_width)
                if bin_idx < len(binned_resp):
                    binned_resp[bin_idx] += r
                    count[bin_idx] += 1
        
        # Compute mean, avoiding division by zero
        mask = count > 0
        binned_resp[mask] = binned_resp[mask] / count[mask]
        
        # Plot
        plt.plot(bins[:-1], binned_resp, label=name)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title(f'Cell {cell_idx} (Prefers {stimulus_names[pref]})')
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('ΔF/F')
    plt.legend()

plt.tight_layout()
plt.savefig('explore/example_cells_by_preference.png')