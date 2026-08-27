import os
from huggingface_hub import hf_hub_download

def download_batch():
    repo_id = "alan-turing-institute/turing-synthetic-radar-dataset"
    cache_dir = "data/raw"
    
    print("==================================================")
    print("PHASE 12: DOWNLOADING DATASET BATCH (12 FILES)")
    print("==================================================")
    
    downloaded_files = []
    
    for i in range(12):
        filename = f"scan/train_scan/config_{i}.h5"
        print(f"Downloading {filename}...")
        try:
            filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir
            )
            downloaded_files.append(filepath)
            print(f"  -> Success: {filepath}")
        except Exception as e:
            print(f"  -> Failed: {e}")
            
    print("\nBatch download complete.")
    return downloaded_files

if __name__ == "__main__":
    download_batch()
