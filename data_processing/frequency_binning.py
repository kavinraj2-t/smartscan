import numpy as np

def assign_frequency_bands(frequencies, num_bands=10):
    freq_min = np.min(frequencies)
    freq_max = np.max(frequencies)
    
    # np.linspace gives num_bands + 1 edges
    edges = np.linspace(freq_min, freq_max, num_bands + 1)
    edges[-1] += 1e-5  # Ensure max value falls into the last bin
    
    bands = np.digitize(frequencies, edges) - 1
    bands = np.clip(bands, 0, num_bands - 1)
    
    return bands, freq_min, freq_max, edges
