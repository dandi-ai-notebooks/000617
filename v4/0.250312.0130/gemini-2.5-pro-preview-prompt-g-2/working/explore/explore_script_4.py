# explore_script_4.py
# This script loads the NWB file and plots dF/F traces for a few selected ROIs.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get DFF traces and corresponding table
dff_traces = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces']
dff_data = dff_traces.data
dff_timestamps = dff_traces.timestamps[:] # Load all timestamps for this selection

# Get cell specimen table to find valid ROIs and their IDs
cell_table_df = nwb.processing['ophys']['image_segmentation'].plane_segmentations['cell_specimen_table'].to_dataframe()

# Select a few ROIs
num_rois_to_plot = 3
time_points_to_plot = 500 # approx 50 seconds of data at 10 Hz

valid_rois_df = cell_table_df[cell_table_df['valid_roi'] == True]

if len(valid_rois_df) >= num_rois_to_plot:
    selected_rois_df = valid_rois_df.head(num_rois_to_plot)
    print(f"Plotting first {num_rois_to_plot} valid ROIs.")
else:
    selected_rois_df = cell_table_df.head(num_rois_to_plot)
    print(f"Not enough valid ROIs found, plotting first {num_rois_to_plot} ROIs from the table.")

roi_indices = selected_rois_df.index.to_list() # These are indices IN THE ORIGINAL cell_table_df
# The dff_data is indexed 0 to N-1 for ROIs. We need to map selected_rois_df.index
# (which are original IDs if .id is used, or 0-based if .reset_index() was used on cell_table_df)
# to the column index in dff_data.
# The dff_traces.rois are DynamicTableRegion linking to cell_table.
# So, the indices of selected_rois_df (if it's a slice of cell_table_df) should correspond to columns in dff_data.
# If cell_table_df.index are the original IDs (like 1285902696), we need to find their 0-based index in the table.
# Let's assume cell_table_df.index are 0-based indices into the original table that dff_data columns correspond to.
# This should be correct as .to_dataframe() usually gives a 0-indexed DataFrame unless 'id' is set as index.
# The nwb-file-info shows cell_specimen_table.id so .to_dataframe() will use that as index.
# We need the positional indices of these IDs within the original table order.
# A robust way: get all IDs from cell_table, find positions of our selected IDs.
all_roi_ids_in_table_order = nwb.processing['ophys']['image_segmentation'].plane_segmentations['cell_specimen_table'].id[:]
selected_roi_actual_ids = selected_rois_df.index # These are the actual IDs from the 'id' column
# Find the 0-based indices for dff_data
selected_column_indices = [np.where(all_roi_ids_in_table_order == roi_id)[0][0] for roi_id in selected_roi_actual_ids]

fig, axes = plt.subplots(num_rois_to_plot, 1, figsize=(12, 2 * num_rois_to_plot), sharex=True)
if num_rois_to_plot == 1: # Make axes iterable if only one subplot
    axes = [axes]

for i, (original_id, col_idx) in enumerate(zip(selected_roi_actual_ids, selected_column_indices)):
    trace = dff_data[:time_points_to_plot, col_idx]
    timestamps_subset = dff_timestamps[:time_points_to_plot]
    axes[i].plot(timestamps_subset, trace)
    cell_specimen_id = selected_rois_df.loc[original_id, 'cell_specimen_id']
    valid_status = selected_rois_df.loc[original_id, 'valid_roi']
    axes[i].set_title(f"ROI ID (original): {original_id}, Cell Specimen ID: {cell_specimen_id}, Valid: {valid_status}")
    axes[i].set_ylabel("dF/F")

axes[-1].set_xlabel("Time (s)")
plt.suptitle(f"dF/F Traces for {num_rois_to_plot} Selected ROIs (First {time_points_to_plot} samples)")
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.savefig('explore/dff_traces.png')
plt.close(fig)

print(f"Selected ROI IDs (original): {selected_roi_actual_ids.to_list()}")
print(f"Corresponding column indices in dff_data: {selected_column_indices}")
print("dF/F traces plot saved to explore/dff_traces.png")

io.close()