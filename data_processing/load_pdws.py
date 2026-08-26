import h5py
import numpy as np
import os

def load_tsrd_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
        feature_names = [x.decode('utf-8') for x in f['metadata']['feature_names']]
    
    pdw_dict = {
        name: data[:, i] for i, name in enumerate(feature_names)
    }
    return pdw_dict
