# explore_script_1.py
# This script loads the NWB file and prints basic information,
# including the cell_specimen_table as a pandas DataFrame.

import pynwb
import h5py
import remfile
import pandas as pd

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"NWB File Identifier: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")

# Print cell_specimen_table
cell_specimen_table_df = nwb.processing['ophys']['image_segmentation'].plane_segmentations['cell_specimen_table'].to_dataframe()
print("\\nCell Specimen Table:")
print(cell_specimen_table_df.head())

# Print imaging plane info
imaging_plane = nwb.imaging_planes['imaging_plane_1']
print("\\nImaging Plane Info:")
print(f"  Description: {imaging_plane.description}")
print(f"  Location: {imaging_plane.location}")
print(f"  Indicator: {imaging_plane.indicator}")
print(f"  Excitation Lambda: {imaging_plane.excitation_lambda}")
print(f"  Imaging Rate: {imaging_plane.imaging_rate}")

io.close()