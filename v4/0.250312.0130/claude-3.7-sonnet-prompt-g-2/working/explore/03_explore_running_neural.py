'''
This script explores the relationship between running behavior and neural activity.
It analyzes whether running speed correlates with changes in neural activity.
'''

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import signal
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/27dd7936-b3e7-45af-aca0-dc98b5954d19/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get running speed data
running_module = nwb.processing['running']
speed = running_module.data_interfaces['speed']
speed_data = speed.data[:]  # Running speed in cm/s
speed_times = speed.timestamps[:]  # Timestamps for running speed

# Get neural data (DF/F)
ophys = nwb.processing['ophys']
dff = ophys.data_interfaces['dff']
roi_response_series = dff.roi_response_series['traces']
dff_data = roi_response_series.data[:]  # shape: (timepoints, ROIs)
dff_times = roi_response_series.timestamps[:]  # Timestamps for DF/F

print(f"Speed data shape: {speed_data.shape}")
print(f"DFF data shape: {dff_data.shape}")
print(f"Speed time range: {speed_times[0]} to {speed_times[-1]} seconds")
print(f"DFF time range: {dff_times[0]} to {dff_times[-1]} seconds")

# Plot running speed over time
plt.figure(figsize=(15, 5))
plt.plot(speed_times[:10000], speed_data[:10000])  # Plot first 10000 points
plt.xlabel('Time (s)')
plt.ylabel('Running Speed (cm/s)')
plt.title('Running Speed Over Time (First 10000 Timepoints)')
plt.savefig('explore/running_speed.png')

# Determine timepoints with high running speed (top 25%)
high_speed_threshold = np.percentile(speed_data, 75)
high_speed_indices = np.where(speed_data > high_speed_threshold)[0]
print(f"High speed threshold: {high_speed_threshold} cm/s")
print(f"Number of high speed timepoints: {len(high_speed_indices)}")
print(f"Percentage of time running fast: {100 * len(high_speed_indices) / len(speed_data):.2f}%")

# Define periods of high running (consecutive timepoints above threshold)
high_speed_periods = []
if len(high_speed_indices) > 0:
    # Find consecutive indices
    consecutive_periods = np.split(high_speed_indices, 
                                   np.where(np.diff(high_speed_indices) > 1)[0] + 1)
    
    # Filter for periods longer than 1 second (assuming sampling rate)
    sampling_rate = 1 / (speed_times[1] - speed_times[0])
    min_period_length = int(1.0 * sampling_rate)  # 1 second
    
    for period in consecutive_periods:
        if len(period) >= min_period_length:
            start_time = speed_times[period[0]]
            end_time = speed_times[period[-1]]
            duration = end_time - start_time
            if duration >= 1.0:  # At least 1 second
                high_speed_periods.append({
                    'start_idx': period[0],
                    'end_idx': period[-1],
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })

print(f"Number of high speed periods: {len(high_speed_periods)}")
if len(high_speed_periods) > 0:
    durations = [period['duration'] for period in high_speed_periods]
    print(f"Average duration of high speed periods: {np.mean(durations):.2f} seconds")
    print(f"Longest high speed period: {np.max(durations):.2f} seconds")
    print(f"Shortest high speed period: {np.min(durations):.2f} seconds")

# Downsample running speed to match DF/F sampling rate
def downsample_to_target_times(source_times, source_data, target_times):
    """Downsample data to match target timestamps."""
    # For each target timestamp, find the nearest source timestamp
    result = np.zeros(len(target_times))
    for i, t in enumerate(target_times):
        # Find the closest time in source_times
        idx = np.argmin(np.abs(source_times - t))
        result[i] = source_data[idx]
    return result

# Downsample running speed to match DF/F timestamps
downsampled_speed = downsample_to_target_times(speed_times, speed_data, dff_times)

# Calculate correlation between running speed and neural activity
correlations = []
for roi_idx in range(dff_data.shape[1]):
    roi_dff = dff_data[:, roi_idx]
    corr, p_val = pearsonr(downsampled_speed, roi_dff)
    correlations.append({
        'roi_idx': roi_idx,
        'correlation': corr,
        'p_value': p_val
    })

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

# Print top 10 correlated ROIs
print("\nTop 10 ROIs correlated with running speed:")
for i, corr_data in enumerate(correlations[:10]):
    print(f"ROI {corr_data['roi_idx']}: r = {corr_data['correlation']:.3f}, p = {corr_data['p_value']:.3e}")

# Plot correlation distribution
plt.figure(figsize=(10, 6))
corr_values = [c['correlation'] for c in correlations]
plt.hist(corr_values, bins=20)
plt.xlabel('Pearson Correlation with Running Speed')
plt.ylabel('Number of ROIs')
plt.title('Distribution of Neural Activity Correlations with Running Speed')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig('explore/running_correlation_histogram.png')

# Plot the most positively and negatively correlated ROIs with running speed
plt.figure(figsize=(15, 10))

# Top 3 positively correlated
top_pos_indices = [correlations[i]['roi_idx'] for i in range(3)]
for i, roi_idx in enumerate(top_pos_indices):
    plt.subplot(3, 2, 2*i+1)
    plt.plot(dff_times[:1000], dff_data[:1000, roi_idx], 'b-', label='DF/F')
    plt.title(f'ROI {roi_idx} (r = {correlations[i]["correlation"]:.3f})')
    plt.ylabel('DF/F')
    
    ax2 = plt.gca().twinx()
    ax2.plot(dff_times[:1000], downsampled_speed[:1000], 'r-', alpha=0.5, label='Speed')
    ax2.set_ylabel('Speed (cm/s)', color='r')
    plt.xlabel('Time (s)')

# Top 3 negatively correlated
neg_corr_sorted = sorted(correlations, key=lambda x: x['correlation'])
top_neg_indices = [neg_corr_sorted[i]['roi_idx'] for i in range(3)]
for i, roi_idx in enumerate(top_neg_indices):
    plt.subplot(3, 2, 2*i+2)
    plt.plot(dff_times[:1000], dff_data[:1000, roi_idx], 'b-', label='DF/F')
    corr_val = [c['correlation'] for c in correlations if c['roi_idx'] == roi_idx][0]
    plt.title(f'ROI {roi_idx} (r = {corr_val:.3f})')
    plt.ylabel('DF/F')
    
    ax2 = plt.gca().twinx()
    ax2.plot(dff_times[:1000], downsampled_speed[:1000], 'r-', alpha=0.5, label='Speed')
    ax2.set_ylabel('Speed (cm/s)', color='r')
    plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig('explore/running_correlated_cells.png')

# Calculate average neural activity during high running vs. low running periods
if len(high_speed_periods) > 0:
    # Find matching periods in DF/F data
    high_run_dff = []
    for period in high_speed_periods[:10]:  # Take only first 10 periods to avoid memory issues
        # Find DF/F indices corresponding to this time period
        start_idx = np.argmin(np.abs(dff_times - period['start_time']))
        end_idx = np.argmin(np.abs(dff_times - period['end_time']))
        
        if end_idx > start_idx:
            period_dff = dff_data[start_idx:end_idx, :]
            high_run_dff.append(period_dff)
    
    # Calculate average DF/F during high running periods
    if high_run_dff:
        high_run_dff = np.concatenate(high_run_dff, axis=0)
        high_run_avg_dff = np.mean(high_run_dff, axis=0)
        
        # Calculate average DF/F during low running periods
        low_speed_threshold = np.percentile(speed_data, 25)
        low_speed_indices = np.where(speed_data < low_speed_threshold)[0]
        
        # Sample the same number of timepoints from low running periods
        if len(low_speed_indices) > high_run_dff.shape[0]:
            sampled_indices = np.random.choice(low_speed_indices, high_run_dff.shape[0], replace=False)
            
            # Convert to DF/F timepoints
            low_run_times = speed_times[sampled_indices]
            low_run_dff_indices = [np.argmin(np.abs(dff_times - t)) for t in low_run_times]
            low_run_dff = dff_data[low_run_dff_indices, :]
            low_run_avg_dff = np.mean(low_run_dff, axis=0)
            
            # Compare activity
            plt.figure(figsize=(10, 6))
            plt.scatter(high_run_avg_dff, low_run_avg_dff, alpha=0.5)
            plt.plot([-0.2, 0.2], [-0.2, 0.2], 'k--')  # Diagonal line
            plt.xlabel('Average DF/F during high running')
            plt.ylabel('Average DF/F during low running')
            plt.title('Comparison of DF/F during high vs. low running periods')
            plt.grid(True)
            plt.savefig('explore/high_vs_low_running_dff.png')
            
            # Calculate difference and find most modulated cells
            dff_diff = high_run_avg_dff - low_run_avg_dff
            top_increase_idx = np.argsort(dff_diff)[-10:]  # Top 10 cells with higher activity during running
            top_decrease_idx = np.argsort(dff_diff)[:10]   # Top 10 cells with lower activity during running
            
            print("\nTop 10 cells with increased activity during running:")
            for idx in top_increase_idx:
                print(f"ROI {idx}: increase by {dff_diff[idx]:.4f} DF/F units")
            
            print("\nTop 10 cells with decreased activity during running:")
            for idx in top_decrease_idx:
                print(f"ROI {idx}: decrease by {dff_diff[idx]:.4f} DF/F units")