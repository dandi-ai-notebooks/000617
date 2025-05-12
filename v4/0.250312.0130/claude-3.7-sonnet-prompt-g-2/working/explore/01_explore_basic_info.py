'''
This script explores basic information about the NWB file:
- Basic metadata
- Structure and organization
- Available data types and their dimensions
- Some basic statistics about the data
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the NWB file
print("\n=== Basic Information ===")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject genotype: {nwb.subject.genotype}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject sex: {nwb.subject.sex}")

# Print information about available data
print("\n=== Available Data Types ===")
print("Acquisition data:")
for name, data in nwb.acquisition.items():
    print(f"  - {name}: {type(data).__name__}")

print("\nStimulus templates:")
for name, data in nwb.stimulus_template.items():
    print(f"  - {name}: {type(data).__name__}")

print("\nProcessing modules:")
for module_name, module in nwb.processing.items():
    print(f"  - {module_name} ({module.description}):")
    for name, interface in module.data_interfaces.items():
        print(f"    * {name}: {type(interface).__name__}")

print("\nIntervals:")
for name, interval in nwb.intervals.items():
    print(f"  - {name}: {type(interval).__name__}")

# Get information about ROIs
print("\n=== ROI Information ===")
ophys = nwb.processing['ophys']
image_seg = ophys.data_interfaces['image_segmentation']
plane_seg = image_seg.plane_segmentations['cell_specimen_table']
imaging_plane = plane_seg.imaging_plane
print(f"Imaging plane: {imaging_plane.description}")
print(f"Imaging plane location: {imaging_plane.location}")
print(f"Imaging rate: {imaging_plane.imaging_rate} Hz")
print(f"Indicator: {imaging_plane.indicator}")

# Number of ROIs
num_rois = len(plane_seg.id.data[:])
print(f"Number of ROIs: {num_rois}")

# Get fluorescence traces information
print("\n=== Fluorescence Traces Information ===")
dff = ophys.data_interfaces['dff']
roi_response_series = dff.roi_response_series['traces']
trace_data = roi_response_series.data
num_timepoints = trace_data.shape[0]
print(f"Number of timepoints: {num_timepoints}")
print(f"Trace data shape: {trace_data.shape}")
print(f"Sampling rate: approximately {num_timepoints / (roi_response_series.timestamps[-1] - roi_response_series.timestamps[0]):.2f} Hz")

# Get timing of the first few stimulus presentations
print("\n=== Stimulus Presentations ===")
for stim_name in ['gray_presentations', 'movie_clip_A_presentations', 'movie_clip_B_presentations', 'movie_clip_C_presentations']:
    presentations = nwb.intervals[stim_name]
    num_presentations = len(presentations.id.data[:])
    print(f"{stim_name}: {num_presentations} presentations")
    if num_presentations > 0:
        first_start = presentations.start_time.data[0]
        first_stop = presentations.stop_time.data[0]
        first_duration = first_stop - first_start
        print(f"  First presentation: start={first_start:.2f}s, stop={first_stop:.2f}s, duration={first_duration:.2f}s")

# Examine motion correction data
print("\n=== Motion Correction Information ===")
motion_x = ophys.data_interfaces['ophys_motion_correction_x'].data
motion_y = ophys.data_interfaces['ophys_motion_correction_y'].data
print(f"X motion range: {np.min(motion_x)} to {np.max(motion_x)} pixels")
print(f"Y motion range: {np.min(motion_y)} to {np.max(motion_y)} pixels")

# Create a plot of motion correction
plt.figure(figsize=(10, 5))
timestamps = ophys.data_interfaces['ophys_motion_correction_x'].timestamps[:5000]  # Use first 5000 timepoints
plt.plot(timestamps, motion_x[:5000], 'b-', label='X motion')
plt.plot(timestamps, motion_y[:5000], 'r-', label='Y motion')
plt.xlabel('Time (s)')
plt.ylabel('Motion (pixels)')
plt.title('Motion Correction (First 5000 Timepoints)')
plt.legend()
plt.savefig('explore/motion_correction.png')

# Create a plot of the average fluorescence trace
plt.figure(figsize=(12, 6))
# Get DF/F traces for first 10 ROIs and first 1000 timepoints
dff_data = trace_data[:1000, :10]
timestamps = roi_response_series.timestamps[:1000]
for i in range(min(10, dff_data.shape[1])):
    plt.plot(timestamps, dff_data[:, i], label=f'ROI {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F')
plt.title('DF/F Traces for First 10 ROIs (First 1000 Timepoints)')
plt.legend()
plt.savefig('explore/dff_traces.png')