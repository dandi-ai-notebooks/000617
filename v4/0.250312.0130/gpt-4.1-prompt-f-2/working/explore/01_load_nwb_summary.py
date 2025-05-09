# This script checks loading of the selected NWB file and prints summaries of its key data structures.
# It outputs shapes of important arrays (fluorescence, dF/F, times, ROIs, etc.) and sample entries for ROI/cell metadata.
import pynwb
import h5py
import remfile
import numpy as np

url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, "r")
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

print(f"NWB identifier: {nwb.identifier}")
print(f"Session start: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}   Species: {nwb.subject.species}   Genotype: {nwb.subject.genotype}   Sex: {nwb.subject.sex}")
print("")

# ROI/cell metadata summary
roi_table = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces'].rois.table
df_info = roi_table.to_dataframe()
print(f"PlaneSegmentation Table: {df_info.shape[0]} ROIs x {df_info.shape[1]} fields")
print("Columns:", list(df_info.columns))
print("Sample ROI info (first 5 rows):")
print(df_info.head())
print("")

# DFF array shape and timestamps
dff = nwb.processing['ophys'].data_interfaces['dff'].roi_response_series['traces']
print(f"dF/F data shape: {dff.data.shape} (frames x ROIs)")
print(f"dF/F timestamps shape: {dff.timestamps.shape}")
print("First few timestamps:", dff.timestamps[:5])
print("")

# Some raw fluorescence shape
fl = nwb.processing['ophys'].data_interfaces['corrected_fluorescence'].roi_response_series['traces']
print(f"Raw fluorescence data shape: {fl.data.shape} (frames x ROIs)")
print("")

# Event detection info
event_det = nwb.processing['ophys'].data_interfaces['event_detection']
print(f"Event detection data shape: {event_det.data.shape} (frames x ROIs)")

# Show a little of the metadata on cell masks
print("ROI image masks shape:", np.array(roi_table.image_mask).shape)
print("ROI image mask example (first ROI):", np.array(roi_table.image_mask)[0][0:5,0:5])
print("")

# Stimulus interval structure
interval_names = list(nwb.intervals.keys())
print("Interval tables in NWB:", interval_names)
if "movie_clip_A_presentations" in nwb.intervals:
    df_intv = nwb.intervals["movie_clip_A_presentations"].to_dataframe()
    print("movie_clip_A_presentations table shape:", df_intv.shape)
    print(df_intv.head())
print("")

# Some notes on acquisition Timeseries
acq_keys = list(nwb.acquisition.keys())
print("Top-level acquisition keys:", acq_keys)
for k in acq_keys:
    ts = nwb.acquisition[k]
    print(f"{k}: shape={ts.data.shape}, timestamps_shape={ts.timestamps.shape}, unit={ts.unit}")