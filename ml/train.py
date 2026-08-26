# ml/train.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from ml.rf_scan_env import RFScanEnv

def main():
    print("--- Phase 4: Training PPO Agent ---")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize the Gymnasium environment
    env = RFScanEnv(seed=42)
    
    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    
    # Save a checkpoint every 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/logs/',
        name_prefix='ppo_smart_scan'
    )
    
    print("Starting training for 50,000 timesteps...")
    model.learn(total_timesteps=50000, callback=checkpoint_callback)
    
    # Save the final model
    model_path = "models/ppo_smart_scan"
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
