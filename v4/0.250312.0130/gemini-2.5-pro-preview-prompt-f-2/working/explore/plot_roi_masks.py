# This script loads an NWB file and plots the image masks for selected ROIs.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns # Not needed for imshow

# Connect to DANDI and get the NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
with pynwb.NWBHDF5IO(file=h5_file, mode='r') as io:
    nwb = io.read()

    roi_table = nwb.processing['ophys']['dff']['traces'].rois.table
    roi_table_df = roi_table.to_dataframe()

    # Use the same selection logic as in plot_dff_traces.py
    valid_roi_mask_bool = roi_table_df['valid_roi'].to_numpy(dtype=bool)
    valid_roi_positional_indices = np.where(valid_roi_mask_bool)[0].tolist()
    
    num_rois_to_display = 3 # Match the dF/F plot
    
    selected_positional_indices = []
    if len(valid_roi_positional_indices) >= num_rois_to_display:
        selected_positional_indices = valid_roi_positional_indices[:num_rois_to_display]
    elif len(valid_roi_positional_indices) > 0:
        selected_positional_indices = valid_roi_positional_indices
    else:
        num_available_rois = roi_table_df.shape[0]
        selected_positional_indices = list(range(min(num_rois_to_display, num_available_rois)))

    if not selected_positional_indices:
        print("No ROIs selected to display masks.")
        exit()

    selected_id_labels = roi_table_df.index[selected_positional_indices].tolist()
    
    print(f"Selected {len(selected_positional_indices)} ROIs for mask display.")
    print(f"Positional indices: {selected_positional_indices}")
    print(f"ID labels: {selected_id_labels}")

    # Assuming imaging plane is 512x512 from nwb-file-info
    # ImagingPlane description: "(512, 512) field of view in VISp at depth 175 um"
    # However, image_mask shape is (height, width) of the ROI itself.
    # We need to place these masks onto a larger canvas.
    # Let's find the max dimensions from the actual imaging_plane object if possible or default.
    try:
        # imaging_plane_obj = nwb.imaging_planes[roi_table.imaging_plane.name] # This might not be direct
        # For this file, roi_table.imaging_plane is the actual ImagingPlane object
        imaging_plane_obj = roi_table.imaging_plane
        # The grid_spacing attribute gives pixel size and origin_coords gives offset.
        # The actual dimensions (e.g. 512x512) are often part of the description or linked TwoPhotonSeries.
        # The image_mask itself contains its shape. For overlay, we need a common canvas.
        # Let's determine canvas size by max x+width and y+height of selected ROIs or use a default.
        max_x_coord = 0
        max_y_coord = 0
        for id_label in selected_id_labels:
            roi_data = roi_table_df.loc[id_label]
            max_x_coord = max(max_x_coord, roi_data['x'] + roi_data['width'])
            max_y_coord = max(max_y_coord, roi_data['y'] + roi_data['height'])
        
        # Fallback to a default if calculated values are too small (e.g. if ROIs are tiny at 0,0)
        # From NWB info: description: "(512, 512) field of view..."
        canvas_height = int(max(512, max_y_coord + 10)) # Add some padding
        canvas_width = int(max(512, max_x_coord + 10))
        print(f"Determined canvas size: {canvas_height}x{canvas_width}")

    except Exception as e:
        print(f"Could not determine canvas size from imaging plane, defaulting to 512x512. Error: {e}")
        canvas_height, canvas_width = 512, 512

    # Create an overlay image
    # Initialize with zeros
    overlay_image = np.zeros((canvas_height, canvas_width), dtype=float)
    
    roi_details_for_plot = []
    actual_mask_shapes = []

    for i, id_label in enumerate(selected_id_labels):
        roi_data_series = roi_table_df.loc[id_label]
        mask = roi_data_series['image_mask'] # This is a 2D array
        actual_mask_shapes.append(mask.shape)

        # Assuming image_mask is already full-plane (e.g., 512x512)
        # and contains 0s outside the ROI, and 1s (or other values) for the ROI pixels.
        # Also assuming overlay_image and mask have the same dimensions based on canvas_height/width logic.
        if mask.shape == (canvas_height, canvas_width):
            # Use (i+1) to give different ROIs different values for heatmap differentiation
            # if mask is boolean, convert to float (0.0 and 1.0)
            overlay_image = np.maximum(overlay_image, mask.astype(float) * (i + 1))
        else:
            print(f"Warning: ROI {id_label} mask shape {mask.shape} differs from determined canvas {canvas_height}x{canvas_width}. Skipping this ROI for overlay.")
            # Optionally, try to place it if it's smaller using x,y,width,height from table
            # For now, keeping it simple: skip if shapes don't match the initialized overlay_image
            continue
        
        # For text labels, use x, y, width, height from the table, which describe the ROI's bounding box
        roi_x_tl = int(roi_data_series['x'])
        roi_y_tl = int(roi_data_series['y'])
        roi_w = int(roi_data_series['width'])
        roi_h = int(roi_data_series['height'])
        
        cell_id = roi_data_series['cell_specimen_id']
        is_valid = roi_data_series['valid_roi']
        label_text = f"ROI {cell_id if cell_id != -1 else id_label} ({'V' if is_valid else 'NV'})"
        # Place text label at the center of the metadata bounding box
        roi_details_for_plot.append({'label': label_text, 'x': roi_x_tl + roi_w/2, 'y': roi_y_tl + roi_h/2, 'color_val': i + 1})
    
    print(f"Actual mask shapes encountered for selected ROIs: {actual_mask_shapes}")


    plt.figure(figsize=(10, 10))
    if np.max(overlay_image) > 0: # Check if there's anything to plot
        # Using a discrete colormap might be good here if ROIs have distinct integer values
        num_colors = len(selected_id_labels)
        # cmap = plt.cm.get_cmap('viridis', num_colors) if num_colors > 0 else 'viridis' # Deprecated
        cmap = plt.get_cmap('viridis', num_colors) if num_colors > 0 else plt.get_cmap('viridis')
        plt.imshow(overlay_image, cmap=cmap, interpolation='nearest', origin='lower')
        # Add a colorbar that shows which value corresponds to which ROI index
        if num_colors > 0:
            # Ensure ticks match the values assigned (1 to num_colors)
            colorbar_ticks = np.arange(1, num_colors + 1)
            cbar = plt.colorbar(ticks=colorbar_ticks)
            # Create labels for these ticks. ROIs are numbered 1 to num_colors in the overlay_image.
            # The labels should correspond to the actual ROIs plotted.
            # roi_details_for_plot contains {'label': ..., 'color_val': i+1}
            # We want to label tick 'j' with the label of ROI whose color_val was 'j'.
            tick_labels = [""] * num_colors
            for detail in roi_details_for_plot:
                 if 1 <= detail['color_val'] <= num_colors:
                    tick_labels[detail['color_val']-1] = detail['label'].split(" (")[0] # Get "ROI ID" part

            # Filter out empty labels if some color_vals were not in roi_details (should not happen here)
            # Or, more simply, map selected_id_labels or their positional indices to the ticks
            # The ROIs are indexed 0 to num_colors-1 in selected_id_labels
            # Their values in overlay_image are 1 to num_colors
            
            # Simplest: label ticks 1 to N with "ROI 1", "ROI 2", ... corresponding to plotting order
            # Or use the actual labels from roi_details_for_plot if they are concise enough
            cbar_labels = [roi_details_for_plot[k]['label'].split(" (")[0] for k in range(num_colors)] # Assumes roi_details_for_plot is ordered 0 to num_colors-1
            cbar.ax.set_yticklabels(cbar_labels)
    else:
        plt.text(0.5, 0.5, "No mask data to display or masks are empty.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        print("Overlay image is empty or all zeros.")

    # Add text labels for each ROI (optional, can get cluttered)
    # for detail in roi_details_for_plot:
    #     plt.text(detail['x'], detail['y'], detail['label'], color='red', ha='center', va='center', fontsize=8)

    plt.title(f'Overlay of Selected ROI Masks (Colors indicate different ROIs)')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    # plt.gca().invert_yaxis() # Often image origin is top-left
    plt.grid(False) # Grid is usually not helpful for images
    plt.savefig('explore/roi_masks.png')
    plt.close()

    print(f"Saved ROI masks plot to explore/roi_masks.png")