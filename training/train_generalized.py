import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv
from environment.normalization import compute_normalization_stats

def train_generalized():
    print("==================================================")
    print("PHASE 11: GENERALIZED MULTI-FILE TRAINING")
    print("==================================================")
    
    # Define training files: 10 files total (0-5 and 7-10)
    base_dir = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan"
    train_files = [os.path.join(base_dir, f"config_{i}.h5") for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]]
    
    # 1. Calculate global normalization statistics across all 10 files
    norm_path = "environment/norm_generalized.json"
    compute_normalization_stats(train_files, time_window=50000, num_bands=10, output_path=norm_path)
    
    # 2. Initialize Generalized Environment
    # Using Model A (Stochastic Inference will be used, no artificial penalties needed)
    env = RFScanEnv(
        filepaths=train_files,
        time_window=50000,
        num_bands=10,
        enable_exploration=False,
        enable_repeat_penalty=False,
        norm_stats_path=norm_path
    )
    
    env = Monitor(env)
    
    # 3. Train Model
    config_name = "Model_Generalized_10Files"
    tensorboard_log_dir = f"./logs/{config_name}"
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        device="cpu"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/{config_name}_checkpoints/",
        name_prefix="ppo_model"
    )
    
    # 100,000 timesteps to ensure it sees all 10 files sufficiently
    timesteps = 100000 
    
    print(f"\nTraining {config_name} for {timesteps} steps...")
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    
    save_path = f"./models/{config_name}_final"
    model.save(save_path)
    print(f"\nFinished training {config_name}. Model saved to {save_path}.zip\n")

if __name__ == "__main__":
    train_generalized()
