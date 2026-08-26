import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv

def train_model(filepath, config_name, enable_exploration, enable_repeat_penalty, timesteps=50000):
    print(f"\n==================================================")
    print(f"TRAINING CONFIGURATION: {config_name}")
    print(f"Exploration Bonus: {enable_exploration}")
    print(f"Repeat Penalty:    {enable_repeat_penalty}")
    print(f"==================================================")
    
    env = RFScanEnv(
        filepath=filepath,
        enable_exploration=enable_exploration,
        enable_repeat_penalty=enable_repeat_penalty
    )
    
    # Wrap with Monitor to get episode statistics natively in tensorboard
    env = Monitor(env)
    
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
    
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    
    save_path = f"./models/{config_name}_final"
    model.save(save_path)
    print(f"Finished training {config_name}. Model saved to {save_path}.zip\n")

if __name__ == "__main__":
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    
    # Phase 6 & 9: Ablation Training
    # Timesteps set to 20,000 to keep runtimes manageable for demonstration
    
    # Train Model A (No Exploration Bonus, No Penalty)
    train_model(filepath, "Model_A_NoExp_NoPen", False, False, timesteps=20000)
    
    # Train Model B (Exploration Bonus Only)
    train_model(filepath, "Model_B_ExpOnly", True, False, timesteps=20000)
    
    # Train Model C (Exploration Bonus + Repeat Penalty)
    train_model(filepath, "Model_C_Full", True, True, timesteps=20000)
