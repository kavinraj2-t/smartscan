# data_processing/inspect_dataset.py
import os
import h5py
import numpy as np
from huggingface_hub import hf_hub_download

def inspect_file(filepath):
    print(f"\n--- Inspecting {filepath} ---")
    with h5py.File(filepath, 'r') as f:
        print("Root keys:", list(f.keys()))
        
        for key in f.keys():
            print(f"\nDataset: '{key}'")
            ds = f[key]
            
            if isinstance(ds, h5py.Group):
                print("  Type: Group")
                print("  Subkeys:", list(ds.keys()))
                continue
                
            print(f"  Shape: {ds.shape}")
            print(f"  Dtype: {ds.dtype}")
            
            # Print attributes / metadata
            if ds.attrs:
                print("  Metadata (Attributes):")
                for attr_key, attr_val in ds.attrs.items():
                    print(f"    {attr_key}: {attr_val}")
                    
            if len(ds.shape) > 0 and ds.shape[0] > 0:
                print(f"  Total records: {ds.shape[0]}")
                print("  Sample rows (first 3):")
                print(ds[:3])
                
                # If structured array (like PDWs often are)
                if ds.dtype.names:
                    print("  Fields:", ds.dtype.names)
                    if 'ToA' in ds.dtype.names or 'toa' in ds.dtype.names:
                        toa_col = 'ToA' if 'ToA' in ds.dtype.names else 'toa'
                        print(f"  Min {toa_col}: {np.min(ds[toa_col])}")
                        print(f"  Max {toa_col}: {np.max(ds[toa_col])}")
                    if 'Frequency' in ds.dtype.names or 'frequency' in ds.dtype.names or 'freq' in ds.dtype.names:
                        freq_col = 'Frequency' if 'Frequency' in ds.dtype.names else ('frequency' if 'frequency' in ds.dtype.names else 'freq')
                        print(f"  Min Frequency: {np.min(ds[freq_col])}")
                        print(f"  Max Frequency: {np.max(ds[freq_col])}")

def main():
    repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"
    filename = "scan/train_scan/config_0.h5"
    
    print("Attempting to download sample file from Hugging Face...")
    try:
        # Check if HF_TOKEN is in environment, otherwise it might fail if gated
        filepath = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir="data/raw"
        )
        print(f"Successfully downloaded to {filepath}")
        inspect_file(filepath)
    except Exception as e:
        print(f"Failed to download from Hugging Face. Error: {e}")
        print("\nNote: The TSRD dataset might require gated access.")
        print("Please authenticate using 'hf auth login' or download a sample file manually.")

if __name__ == "__main__":
    main()
