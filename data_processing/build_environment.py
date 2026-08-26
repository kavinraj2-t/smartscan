import numpy as np
import os
import sys

# Add parent directory to path to allow running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.load_pdws import load_tsrd_file
from data_processing.frequency_binning import assign_frequency_bands

def calculate_toa_statistics(toas):
    diffs = np.diff(toas)
    stats = {
        "mean_diff": np.mean(diffs),
        "median_diff": np.median(diffs),
        "p90_diff": np.percentile(diffs, 90),
        "total_duration": toas[-1] - toas[0]
    }
    return diffs, stats

def build_environment(filepath, time_window=50000, num_bands=10):
    pdws = load_tsrd_file(filepath)
    toas = pdws['ToA']
    freqs = pdws['Frequency']
    amps = pdws['Amplitude']
    
    # 1. ToA Statistics
    diffs, toa_stats = calculate_toa_statistics(toas)
    
    # 2. Frequency Binning
    bands, f_min, f_max, edges = assign_frequency_bands(freqs, num_bands=num_bands)
    
    # 3. Time Window Aggregation
    min_toa = toas[0]
    max_toa = toas[-1]
    
    num_windows = int(np.ceil((max_toa - min_toa) / time_window))
    
    gt_pulse_count = np.zeros((num_windows, num_bands), dtype=int)
    gt_mean_amp = np.zeros((num_windows, num_bands), dtype=float)
    gt_max_amp = np.zeros((num_windows, num_bands), dtype=float)
    
    window_indices = np.floor((toas - min_toa) / time_window).astype(int)
    window_indices = np.clip(window_indices, 0, num_windows - 1)
    
    np.add.at(gt_pulse_count, (window_indices, bands), 1)
    
    amp_sums = np.zeros((num_windows, num_bands), dtype=float)
    np.add.at(amp_sums, (window_indices, bands), amps)
    
    mask = gt_pulse_count > 0
    gt_mean_amp[mask] = amp_sums[mask] / gt_pulse_count[mask]
    
    np.maximum.at(gt_max_amp, (window_indices, bands), amps)
    
    gt_signal_present = (gt_pulse_count > 0).astype(int)
    
    avg_pulses_per_window = len(toas) / num_windows
    
    print("\n==================================================")
    print("PHASE 2 — PREPROCESSING RESULTS")
    print("==================================================")
    
    print("\nFREQUENCY BINNING:")
    print(f"  Observed Freq Min: {f_min:.2f} MHz")
    print(f"  Observed Freq Max: {f_max:.2f} MHz")
    print(f"  Configured Bands:  {num_bands}")
    print(f"  Bandwidth per Bin: {(f_max - f_min) / num_bands:.2f} MHz")
    
    print("\nToA STATISTICS:")
    print(f"  Total Duration:    {toa_stats['total_duration']:.2f} units")
    print(f"  Mean Interval:     {toa_stats['mean_diff']:.2f} units")
    print(f"  Median Interval:   {toa_stats['median_diff']:.2f} units")
    print(f"  90th Percentile:   {toa_stats['p90_diff']:.2f} units")
    
    print("\nTIME WINDOW AGGREGATION:")
    print(f"  TIME_WINDOW size:  {time_window}")
    print(f"  Total Pulses:      {len(toas)}")
    print(f"  Total Time Steps:  {num_windows}")
    print(f"  Avg Pulses/Step:   {avg_pulses_per_window:.2f}")
    print("==================================================\n")
    
    ground_truth = {
        'pulse_count': gt_pulse_count,
        'signal_present': gt_signal_present,
        'mean_amplitude': gt_mean_amp,
        'max_amplitude': gt_max_amp,
        'num_windows': num_windows,
        'num_bands': num_bands,
        'time_window': time_window
    }
    
    return ground_truth

if __name__ == "__main__":
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    build_environment(filepath, time_window=50000, num_bands=10)
