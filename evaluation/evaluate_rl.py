import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv

def evaluate_rl_model(filepath, model_path, name, enable_exploration, enable_repeat_penalty):
    env = RFScanEnv(
        filepath=filepath,
        enable_exploration=enable_exploration,
        enable_repeat_penalty=enable_repeat_penalty
    )
    
    if not os.path.exists(model_path):
        print(f"\n{name} Results: Model file not found at {model_path}")
        return
        
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    
    hits = 0
    total_scans = env.max_episode_length
    total_reward = 0.0
    
    gt_signal_present = env.ground_truth['signal_present']
    gt_pulse_count = env.ground_truth['pulse_count']
    total_pulses_available = np.sum(gt_pulse_count)
    total_active_windows = np.sum(gt_signal_present)
    
    captured_pulses = 0
    missed_active_windows = 0
    
    terminated = False
    
    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        
        if gt_signal_present[env.current_time_step, action] == 0:
            active_bands_this_step = np.sum(gt_signal_present[env.current_time_step, :])
            if active_bands_this_step > 0:
                missed_active_windows += 1
                
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if info['is_hit']:
            hits += 1
            captured_pulses += gt_pulse_count[env.current_time_step - 1, action]
            
    hit_rate = (hits / total_scans) * 100
    miss_rate = 100 - hit_rate
    interception_rate = (captured_pulses / total_pulses_available) * 100 if total_pulses_available > 0 else 0
    missed_opp_rate = (missed_active_windows / total_active_windows) * 100 if total_active_windows > 0 else 0
    avg_reward = total_reward / total_scans
    
    print(f"\n{name} Results:")
    print(f"  Hit Rate:                  {hit_rate:.2f}%")
    print(f"  Miss Rate:                 {miss_rate:.2f}%")
    print(f"  Interception Rate:         {interception_rate:.2f}%")
    print(f"  Missed Opportunity Rate:   {missed_opp_rate:.2f}%")
    print(f"  Average Reward per Step:   {avg_reward:.3f}")

def main():
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    print("\n==================================================")
    print("PHASE 7 & 9 — RL MODEL ABLATION EVALUATION")
    print("==================================================")
    
    evaluate_rl_model(filepath, "models/Model_A_NoExp_NoPen_final.zip", "Model A (No Exp, No Pen)", False, False)
    evaluate_rl_model(filepath, "models/Model_B_ExpOnly_final.zip", "Model B (Exp Only)", True, False)
    evaluate_rl_model(filepath, "models/Model_C_Full_final.zip", "Model C (Full: Exp + Pen)", True, True)
    
    print("\n==================================================")

if __name__ == "__main__":
    main()
