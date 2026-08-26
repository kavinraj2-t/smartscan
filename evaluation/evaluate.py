import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv
from baselines.round_robin import RoundRobinScheduler
from baselines.random_scan import RandomScheduler
from baselines.adaptive_heuristic import AdaptiveHeuristicScheduler

def evaluate_scheduler(env, scheduler, name):
    obs, info = env.reset()
    
    hits = 0
    total_scans = env.max_episode_length
    total_reward = 0.0
    
    # Ground truth reference for total possible intercepts
    gt_signal_present = env.ground_truth['signal_present']
    gt_pulse_count = env.ground_truth['pulse_count']
    total_pulses_available = np.sum(gt_pulse_count)
    total_active_windows = np.sum(gt_signal_present)
    
    captured_pulses = 0
    missed_active_windows = 0
    
    terminated = False
    
    while not terminated:
        action = scheduler.get_action(obs)
        
        # Check if the chosen band actually had signal (for missed opportunity metric)
        if gt_signal_present[env.current_time_step, action] == 0:
            # We missed an opportunity if any other band was active
            active_bands_this_step = np.sum(gt_signal_present[env.current_time_step, :])
            if active_bands_this_step > 0:
                missed_active_windows += 1
                
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if info['is_hit']:
            hits += 1
            # In our simulation, if we hit a band, we capture its pulses for that time step
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
    
    return {
        "name": name,
        "hit_rate": hit_rate,
        "interception_rate": interception_rate,
        "avg_reward": avg_reward
    }

def main():
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    print("Initializing Environment...")
    env = RFScanEnv(filepath)
    num_bands = env.num_bands
    
    print("\n==================================================")
    print("PHASE 5 — BASELINE ALGORITHM EVALUATION")
    print("==================================================")
    
    schedulers = [
        (RoundRobinScheduler(num_bands), "Round Robin"),
        (RandomScheduler(num_bands), "Random Scan"),
        (AdaptiveHeuristicScheduler(num_bands), "Adaptive Heuristic")
    ]
    
    for scheduler, name in schedulers:
        evaluate_scheduler(env, scheduler, name)
        
    print("\n==================================================")

if __name__ == "__main__":
    main()
