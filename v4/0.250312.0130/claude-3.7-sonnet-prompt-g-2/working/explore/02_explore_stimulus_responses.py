'''
This script explores the relationship between neural activity and stimulus presentations.
It identifies when each type of stimulus was presented and examines neural responses
during those periods.
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get stimulus presentation times
def get_presentation_df(intervals_name):
    """Convert stimulus interval data to a DataFrame."""
    interval = nwb.intervals[intervals_name]
    data = {
        'start_time': interval.start_time.data[:],
        'stop_time': interval.stop_time.data[:],
        'stimulus_name': interval.stimulus_name.data[:],
        'stimulus_block': interval.stimulus_block.data[:] 
    }
    return pd.DataFrame(data)

# Get all stimulus presentation times
stim_types = ['gray', 'movie_clip_A', 'movie_clip_B', 'movie_clip_C']
presentation_dfs = {}
for stim_type in stim_types:
    presentation_dfs[stim_type] = get_presentation_df(f"{stim_type}_presentations")
    
# Print first few presentation times for each stimulus type
print("\n=== First few presentation times ===")
for stim_type, df in presentation_dfs.items():
    print(f"\n{stim_type}:")
    print(df.head())
    print(f"Total presentations: {len(df)}")
    
# Get DF/F traces
ophys = nwb.processing['ophys']
dff = ophys.data_interfaces['dff']
roi_response_series = dff.roi_response_series['traces']
dff_data = roi_response_series.data[:, :]  # shape: (timepoints, ROIs)
timestamps = roi_response_series.timestamps[:]

# Get cell information
image_seg = ophys.data_interfaces['image_segmentation']
plane_seg = image_seg.plane_segmentations['cell_specimen_table']
valid_roi = plane_seg['valid_roi'].data[:]
cell_specimen_ids = plane_seg['cell_specimen_id'].data[:]

# Only look at valid ROIs
valid_indices = np.where(valid_roi == 1)[0]
print(f"\nNumber of valid ROIs: {len(valid_indices)} out of {len(valid_roi)}")

# Function to align neural data to stimulus presentations
def get_aligned_responses(presentation_df, window=(-0.5, 2.5), bin_size=0.1):
    """
    Align neural data to stimulus presentations.
    
    Args:
        presentation_df: DataFrame with stimulus presentation times
        window: Time window around stimulus onset (in seconds)
        bin_size: Time bin size for neural data (in seconds)
        
    Returns:
        aligned_data: Array of shape (n_presentations, n_cells, n_timepoints)
    """
    # Get stimulus onset times (first 100 for demonstration)
    onset_times = presentation_df['start_time'].values[:100]
    
    # Calculate number of time bins
    n_bins = int((window[1] - window[0]) / bin_size)
    
    # Initialize aligned data array
    aligned_data = np.zeros((len(onset_times), len(valid_indices), n_bins))
    
    # For each stimulus presentation
    for i, onset_time in enumerate(onset_times):
        # Get time window around stimulus onset
        window_start = onset_time + window[0]
        window_end = onset_time + window[1]
        
        # Find indices of timestamps within the window
        idx = np.where((timestamps >= window_start) & (timestamps <= window_end))[0]
        
        if len(idx) > 0:
            # Bin the data
            binned_data = np.zeros((len(valid_indices), n_bins))
            for j, t_idx in enumerate(range(0, len(idx), max(1, int(bin_size / (timestamps[1] - timestamps[0]))))):
                if j < n_bins:
                    if t_idx < len(idx):
                        bin_indices = idx[t_idx:min(t_idx + int(bin_size / (timestamps[1] - timestamps[0])), len(idx))]
                        if len(bin_indices) > 0:
                            binned_data[:, j] = np.mean(dff_data[bin_indices, :][:, valid_indices], axis=0)
            
            aligned_data[i, :, :] = binned_data
    
    return aligned_data

# Get aligned responses for each stimulus type (limit to 100 presentations for speed)
print("\nAligning neural responses to stimulus presentations...")
aligned_responses = {}
for stim_type in stim_types:
    print(f"Processing {stim_type}...")
    aligned_responses[stim_type] = get_aligned_responses(presentation_dfs[stim_type])

# Calculate average response across presentations for each cell
avg_responses = {}
for stim_type in stim_types:
    avg_responses[stim_type] = np.mean(aligned_responses[stim_type], axis=0)

# Plot average response across cells for each stimulus type
plt.figure(figsize=(12, 8))
window = (-0.5, 2.5)
bin_size = 0.1
n_bins = int((window[1] - window[0]) / bin_size)
time_bins = np.linspace(window[0], window[1], n_bins)

for i, stim_type in enumerate(stim_types):
    plt.subplot(2, 2, i+1)
    # Mean across all cells
    avg_resp = np.mean(avg_responses[stim_type], axis=0)
    plt.plot(time_bins, avg_resp)
    plt.axvline(x=0, color='r', linestyle='--')  # Stimulus onset
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('ΔF/F')
    plt.title(f'Average response to {stim_type}')

plt.tight_layout()
plt.savefig('explore/avg_responses_by_stimulus.png')

# Plot heatmap of responses for top cells for each stimulus type
def get_responsive_cells(stim_type, n_cells=20):
    """Get indices of most responsive cells for a stimulus type."""
    # Calculate response amplitude (max - baseline)
    baseline_window = np.where((time_bins >= -0.5) & (time_bins <= 0))[0]
    response_window = np.where((time_bins >= 0) & (time_bins <= 1.0))[0]
    
    baseline = np.mean(avg_responses[stim_type][:, baseline_window], axis=1)
    response = np.max(avg_responses[stim_type][:, response_window], axis=1)
    amplitude = response - baseline
    
    # Get indices of top cells
    top_indices = np.argsort(amplitude)[-n_cells:]
    
    return top_indices

# Create custom colormap (white to blue)
colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
cmap_name = 'white_blue'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# Plot heatmap for each stimulus type
for stim_type in stim_types:
    plt.figure(figsize=(10, 8))
    top_cells = get_responsive_cells(stim_type)
    responses = avg_responses[stim_type][top_cells, :]
    
    # Get cell IDs
    cell_ids = cell_specimen_ids[valid_indices[top_cells]]
    
    # Plot heatmap
    plt.imshow(responses, aspect='auto', cmap=cm, 
               extent=[window[0], window[1], 0, len(top_cells)])
    plt.colorbar(label='ΔF/F')
    plt.axvline(x=0, color='r', linestyle='--')  # Stimulus onset
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Cell #')
    plt.title(f'Top {len(top_cells)} cells responding to {stim_type}')
    
    plt.tight_layout()
    plt.savefig(f'explore/{stim_type}_response_heatmap.png')

# Plot single cell responses to different stimuli
# Choose a few cells that respond well to at least one stimulus
all_responsive_cells = set()
for stim_type in stim_types:
    all_responsive_cells.update(get_responsive_cells(stim_type, n_cells=5))

all_responsive_cells = list(all_responsive_cells)[:5]  # Take at most 5

plt.figure(figsize=(15, 10))
for i, cell_idx in enumerate(all_responsive_cells):
    plt.subplot(len(all_responsive_cells), 1, i+1)
    
    for stim_type in stim_types:
        plt.plot(time_bins, avg_responses[stim_type][cell_idx, :], label=stim_type)
    
    plt.axvline(x=0, color='k', linestyle='--')  # Stimulus onset
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('ΔF/F')
    plt.title(f'Cell {cell_specimen_ids[valid_indices[cell_idx]]} responses')
    plt.legend()

plt.tight_layout()
plt.savefig('explore/single_cell_responses.png')

# Plot a spatial map of responsive cells
plt.figure(figsize=(10, 8))

# Get cell spatial information
cell_x = plane_seg['x'].data[valid_indices]
cell_y = plane_seg['y'].data[valid_indices]

# Calculate responsiveness for each stimulus type
cell_responses = {}
for stim_type in stim_types:
    # Calculate response amplitude (max - baseline)
    baseline_window = np.where((time_bins >= -0.5) & (time_bins <= 0))[0]
    response_window = np.where((time_bins >= 0) & (time_bins <= 1.0))[0]
    
    baseline = np.mean(avg_responses[stim_type][:, baseline_window], axis=1)
    response = np.max(avg_responses[stim_type][:, response_window], axis=1)
    amplitude = response - baseline
    
    # Normalize amplitudes to 0-1
    if np.max(amplitude) > 0:
        amplitude = amplitude / np.max(amplitude)
    
    cell_responses[stim_type] = amplitude

# Plot spatial map
plt.scatter(cell_x, cell_y, s=5, c='gray', alpha=0.3)

for stim_type, color in zip(stim_types, ['blue', 'green', 'red', 'purple']):
    # Get responsive cells (with amplitude > 0.5)
    responsive_cells = np.where(cell_responses[stim_type] > 0.5)[0]
    plt.scatter(
        cell_x[responsive_cells], 
        cell_y[responsive_cells], 
        s=50 * cell_responses[stim_type][responsive_cells], 
        c=color, 
        alpha=0.7,
        label=f"{stim_type} responsive"
    )

plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.title('Spatial map of responsive cells')
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

plt.savefig('explore/responsive_cells_spatial_map.png')