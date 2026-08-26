import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv
from baselines.random_scan import RandomScheduler

def run_agent_and_record(env, agent, is_rl=False):
    obs, info = env.reset()
    actions = []
    hits = []
    
    terminated = False
    time_steps = []
    
    while not terminated:
        if is_rl:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = agent.get_action(obs)
            
        time_steps.append(env.current_time_step)
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        hits.append(info['is_hit'])
        
    return np.array(time_steps), np.array(actions), np.array(hits)

def visualize_scan_behavior(filepath, rl_model_path):
    print("Loading Environment...")
    env = RFScanEnv(filepath, enable_exploration=False, enable_repeat_penalty=False)
    
    print("Loading Models...")
    rl_agent = PPO.load(rl_model_path)
    random_agent = RandomScheduler(env.num_bands)
    
    print("Running RL Agent...")
    rl_times, rl_actions, rl_hits = run_agent_and_record(env, rl_agent, is_rl=True)
    
    print("Running Random Agent...")
    env.reset() # Reset env before second run
    rand_times, rand_actions, rand_hits = run_agent_and_record(env, random_agent, is_rl=False)
    
    print("Plotting Data...")
    gt_pulses = env.ground_truth['pulse_count'] # Shape: (time, bands)
    
    # We transpose for imshow so X=time, Y=bands
    heatmap_data = gt_pulses.T
    
    # Log scale the heatmap for better visibility of faint pulses
    heatmap_data = np.log1p(heatmap_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
    
    cmap = plt.cm.Blues
    
    for ax, title, times, actions, hits in zip(axes, 
                                               ["Model A (RL Agent) Scan Behavior", "Random Scan Behavior"],
                                               [rl_times, rand_times],
                                               [rl_actions, rand_actions],
                                               [rl_hits, rand_hits]):
                                               
        # Draw ground truth heatmap
        im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap=cmap, 
                       extent=[0, env.num_windows, -0.5, env.num_bands - 0.5])
                       
        # Overlay actions
        # We split hits and misses for color coding
        hit_times = times[hits]
        hit_actions = actions[hits]
        
        miss_times = times[~hits]
        miss_actions = actions[~hits]
        
        ax.scatter(miss_times, miss_actions, c='red', marker='x', s=10, alpha=0.5, label='Missed Scan')
        ax.scatter(hit_times, hit_actions, c='lime', marker='o', s=20, edgecolors='black', label='Successful Intercept')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequency Band")
        ax.set_yticks(range(env.num_bands))
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("Time Steps (50,000 units per step)")
    
    plt.tight_layout()
    
    output_path = "evaluation/scan_behavior.png"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved successfully to {output_path}")

if __name__ == "__main__":
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    rl_model_path = "models/Model_A_NoExp_NoPen_final.zip"
    visualize_scan_behavior(filepath, rl_model_path)
