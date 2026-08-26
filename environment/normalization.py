import json
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.build_environment import build_environment

def compute_normalization_stats(filepaths, time_window, num_bands, output_path="environment/normalization_stats.json"):
    print("Computing normalization statistics from training data...")
    if isinstance(filepaths, str):
        filepaths = [filepaths]
        
    global_max_pulse = 0.0
    all_active_amps = []
    
    for filepath in filepaths:
        ground_truth = build_environment(filepath, time_window=time_window, num_bands=num_bands)
        
        pulse_counts = ground_truth['pulse_count']
        mean_amps = ground_truth['mean_amplitude']
        
        local_max = float(np.max(pulse_counts))
        if local_max > global_max_pulse:
            global_max_pulse = local_max
            
        active_mask = pulse_counts > 0
        active_amps = mean_amps[active_mask]
        all_active_amps.extend(active_amps.tolist())
    
    if len(all_active_amps) > 0:
        amp_p1 = float(np.percentile(all_active_amps, 1))
        amp_p99 = float(np.percentile(all_active_amps, 99))
    else:
        amp_p1 = 0.0
        amp_p99 = 1.0
        
    stats = {
        "max_pulse_count": global_max_pulse,
        "amp_p1": amp_p1,
        "amp_p99": amp_p99,
        "time_window": time_window,
        "num_bands": num_bands
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Saved normalization stats to {output_path}")
    print(json.dumps(stats, indent=4))
    return stats

def load_normalization_stats(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Normalization stats missing at {filepath}. Run normalization.py first.")
    with open(filepath, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    compute_normalization_stats(filepath, time_window=50000, num_bands=10)
