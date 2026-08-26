import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv
from environment.normalization import compute_normalization_stats
from baselines.random_scan import RandomScheduler
from evaluation.evaluate_rl import evaluate_rl_model

def evaluate_random_agent(env):
    obs, info = env.reset()
    random_agent = RandomScheduler(env.num_bands)
    
    hits = 0
    total_scans = env.max_episode_length
    
    gt_pulse_count = env.ground_truth['pulse_count']
    total_pulses_available = np.sum(gt_pulse_count)
    captured_pulses = 0
    
    terminated = False
    
    while not terminated:
        action = random_agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info['is_hit']:
            hits += 1
            captured_pulses += gt_pulse_count[env.current_time_step - 1, action]
            
    hit_rate = (hits / total_scans) * 100
    interception_rate = (captured_pulses / total_pulses_available) * 100 if total_pulses_available > 0 else 0
    return hit_rate, interception_rate

def run_sensitivity_analysis():
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    
    configs = [
        {"name": "Base Config", "window": 50000, "bands": 10},
        {"name": "Config 1 (Fast Agile)", "window": 20000, "bands": 10},
        {"name": "Config 2 (Slow Dwell)", "window": 100000, "bands": 10},
        {"name": "Config 3 (Wide Bandwidth)", "window": 50000, "bands": 5},
        {"name": "Config 4 (Narrow Bandwidth)", "window": 50000, "bands": 20},
    ]
    
    results = []
    
    print("\n==================================================")
    print("STARTING SENSITIVITY ANALYSIS")
    print("==================================================\n")
    
    for cfg in configs:
        print(f"--- Running {cfg['name']} (TIME_WINDOW={cfg['window']}, NUM_BANDS={cfg['bands']}) ---")
        
        # 1. Compute and save dynamic normalization stats
        norm_path = f"environment/norm_w{cfg['window']}_b{cfg['bands']}.json"
        compute_normalization_stats([filepath], cfg['window'], cfg['bands'], output_path=norm_path)
        
        # 2. Train RL Agent (Model A equivalent: no exploration bonus, no penalty)
        model_name = f"Sens_w{cfg['window']}_b{cfg['bands']}"
        model_save_path = f"models/{model_name}_final"
        
        print(f"\nTraining RL Agent for {cfg['name']}...")
        env = RFScanEnv(
            filepaths=[filepath],
            time_window=cfg['window'],
            num_bands=cfg['bands'],
            enable_exploration=False,
            enable_repeat_penalty=False,
            norm_stats_path=norm_path
        )
        
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        model.learn(total_timesteps=20000)
        model.save(model_save_path)
        
        # 3. Evaluate RL Agent
        print(f"Evaluating RL Agent for {cfg['name']}...")
        obs, info = env.reset()
        rl_hits = 0
        rl_captured_pulses = 0
        gt_pulse_count = env.ground_truth['pulse_count']
        total_pulses_available = np.sum(gt_pulse_count)
        
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if info['is_hit']:
                rl_hits += 1
                rl_captured_pulses += gt_pulse_count[env.current_time_step - 1, action]
                
        rl_hit_rate = (rl_hits / env.max_episode_length) * 100
        rl_interception_rate = (rl_captured_pulses / total_pulses_available) * 100 if total_pulses_available > 0 else 0
        
        # 4. Evaluate Random Agent
        print(f"Evaluating Random Agent for {cfg['name']}...")
        rand_hit_rate, rand_interception_rate = evaluate_random_agent(env)
        
        results.append({
            "name": cfg['name'],
            "window": cfg['window'],
            "bands": cfg['bands'],
            "rl_ir": rl_interception_rate,
            "rand_ir": rand_interception_rate
        })
        
        print(f"{cfg['name']} Complete: RL IR = {rl_interception_rate:.2f}% | Random IR = {rand_interception_rate:.2f}%\n")
        
    print("\n==================================================")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("==================================================")
    print(f"{'Configuration Name':<30} | {'Window':<8} | {'Bands':<5} | {'RL IR (%)':<10} | {'Random IR (%)':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res['name']:<30} | {res['window']:<8} | {res['bands']:<5} | {res['rl_ir']:<10.2f} | {res['rand_ir']:<10.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_sensitivity_analysis()
