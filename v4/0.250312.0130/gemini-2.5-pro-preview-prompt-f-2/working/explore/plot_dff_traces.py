# This script loads an NWB file and plots the dF/F traces for a few selected ROIs.
# It prioritizes valid ROIs and uses their cell_specimen_id.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Connect to DANDI and get the NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
with pynwb.NWBHDF5IO(file=h5_file, mode='r') as io:
    nwb = io.read()

    # Get dF/F data
    dff_module = nwb.processing['ophys']['dff']
    dff_traces_rs = dff_module['traces'] # Corrected: was dff_traces, now dff_traces_rs
    timestamps = dff_traces_rs.timestamps[:]
    all_dff_data = dff_traces_rs.data[:] # Corrected: was data, now all_dff_data

    roi_table = dff_traces_rs.rois.table
    roi_table_df = roi_table.to_dataframe()
    print("First 5 rows of ROI table:")
    print(roi_table_df.head())

    # Get POSITIONAL indices of valid ROIs from the table
    valid_roi_mask = roi_table_df['valid_roi'].to_numpy(dtype=bool)
    valid_roi_positional_indices = np.where(valid_roi_mask)[0].tolist()
    
    num_rois_to_plot = 3
    rois_data_to_plot = np.array([]) # Initialize as empty numpy array
    roi_labels = []
    plot_title_suffix = ""
    
    # These will be the final POSITIONAL indices used for slicing all_dff_data
    final_selected_positional_indices = []
    # These will be the original index labels (IDs) from roi_table_df for fetching metadata
    final_selected_id_labels = [] # Ensure this is defined for later print statements

    if len(valid_roi_positional_indices) >= num_rois_to_plot:
        print(f"Found {len(valid_roi_positional_indices)} valid ROIs. Plotting the first {num_rois_to_plot} based on their order in the table.")
        final_selected_positional_indices = valid_roi_positional_indices[:num_rois_to_plot]
    elif len(valid_roi_positional_indices) > 0:
        print(f"Found {len(valid_roi_positional_indices)} valid ROIs. Plotting all of them.")
        final_selected_positional_indices = valid_roi_positional_indices
    else:
        print(f"No valid ROIs found or fewer than {num_rois_to_plot} available. Plotting the first up to {num_rois_to_plot} ROIs by positional index.")
        num_available_rois = all_dff_data.shape[1]
        final_selected_positional_indices = list(range(min(num_rois_to_plot, num_available_rois)))

    if not final_selected_positional_indices and all_dff_data.shape[1] > 0: # If list is empty but data exists, maybe take first one
         print(f"No specific ROIs selected by criteria, but data exists. Defaulting to first ROI if available.")
         if all_dff_data.shape[1] > 0:
            final_selected_positional_indices = [0] # Take the very first one as a fallback
         else:
            print("No ROIs found in the data at all.")
            exit() # Exit if no data columns
    elif not final_selected_positional_indices and all_dff_data.shape[1] == 0:
        print("No ROIs found in the data at all.")
        exit()


    rois_data_to_plot = all_dff_data[:, final_selected_positional_indices]
    # Get the original index labels (IDs) corresponding to these positional indices
    final_selected_id_labels = roi_table_df.index[final_selected_positional_indices].tolist()

    roi_labels = []
    for pos_idx, id_label in zip(final_selected_positional_indices, final_selected_id_labels):
        cell_id = roi_table_df.loc[id_label, 'cell_specimen_id']
        is_valid = roi_table_df.loc[id_label, 'valid_roi']
        if is_valid:
            roi_labels.append(f"ROI {cell_id if cell_id != -1 else id_label} (Valid)")
        else:
            roi_labels.append(f"ROI {cell_id if cell_id != -1 else id_label} (Not Valid)")
    
    # Determine plot title suffix based on selection
    if len(valid_roi_positional_indices) >= num_rois_to_plot and valid_roi_positional_indices:
        plot_title_suffix = f"First {len(final_selected_positional_indices)} Selected Valid ROIs"
    elif len(valid_roi_positional_indices) > 0:
        plot_title_suffix = f"All {len(final_selected_positional_indices)} Selected Valid ROIs"
    else: # No valid ROIs were prioritized, or not enough
        plot_title_suffix = f"First {len(final_selected_positional_indices)} ROIs by Index (Validity indicated in legend)"
        
    # Create the plot
    sns.set_theme()
    plt.figure(figsize=(15, 8)) # Increased height for better label visibility if many traces
    
    # Check if rois_data_to_plot is 1D (single ROI selected) or 2D
    if rois_data_to_plot.ndim == 1:
        plt.plot(timestamps, rois_data_to_plot, label=roi_labels[0])
    else:
        for i in range(rois_data_to_plot.shape[1]):
            plt.plot(timestamps, rois_data_to_plot[:, i], label=roi_labels[i])

    plt.xlabel('Time (s)')
    plt.ylabel('dF/F')
    plt.title(f'dF/F Traces: {plot_title_suffix}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('explore/dff_traces.png')
    # plt.show() # Do not show, as it will hang
    plt.close()

    print(f"Saved dF/F traces plot to explore/dff_traces.png")
    print(f"Plotted ROIs labels: {roi_labels}")
    if final_selected_id_labels: # Check if the list is not empty
        print(f"Selected ROI original ID labels from table: {final_selected_id_labels}")
        print(f"Corresponding positional indices used for data slicing: {final_selected_positional_indices}")
        print("Details for selected ROIs (using original ID labels):")
        print(roi_table_df.loc[final_selected_id_labels, ['cell_specimen_id', 'valid_roi', 'x', 'y', 'width', 'height']])