import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv
from baselines.random_scan import RandomScheduler
from baselines.round_robin import RoundRobinScheduler
from baselines.adaptive_heuristic import AdaptiveHeuristicScheduler

def run_evaluation(env, agent, is_rl=False, stochastic=False):
    obs, info = env.reset()
    if hasattr(agent, 'reset'):
        agent.reset()
        
    hits = 0
    total_scans = env.max_episode_length
    total_reward = 0.0
    
    gt_signal_present = env.ground_truth['signal_present']
    gt_pulse_count = env.ground_truth['pulse_count']
    total_pulses_available = np.sum(gt_pulse_count)
    total_active_windows = np.sum(gt_signal_present)
    
    captured_pulses = 0
    missed_active_windows = 0
    
    # Tracking for new metrics
    actions_taken = []
    consecutive_scans_list = []
    
    # Simple Interception Delay Tracker
    active_start_times = np.full(env.num_bands, -1)
    delays = []
    
    terminated = False
    
    while not terminated:
        if is_rl:
            action, _ = agent.predict(obs, deterministic=not stochastic)
            action = int(action)
        else:
            action = agent.get_action(obs)
            
        actions_taken.append(action)
        consecutive_scans_list.append(env.consecutive_scan_count)
        
        # Track Missed Opportunities and Delays
        for b in range(env.num_bands):
            is_active = gt_signal_present[env.current_time_step, b]
            
            if is_active and active_start_times[b] == -1:
                active_start_times[b] = env.current_time_step
                
            if not is_active and active_start_times[b] != -1:
                # Burst ended without interception
                missed_active_windows += 1
                active_start_times[b] = -1
                
        # Intercepted!
        if gt_signal_present[env.current_time_step, action]:
            if active_start_times[action] != -1:
                delay = env.current_time_step - active_start_times[action]
                delays.append(delay)
                active_start_times[action] = -1 # Reset for next burst
                
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if info['is_hit']:
            hits += 1
            captured_pulses += gt_pulse_count[env.current_time_step - 1, action]
            
    hit_rate = (hits / total_scans) * 100
    interception_rate = (captured_pulses / total_pulses_available) * 100 if total_pulses_available > 0 else 0
    missed_opp_rate = (missed_active_windows / total_active_windows) * 100 if total_active_windows > 0 else 0
    avg_reward = total_reward / total_scans
    
    avg_delay = np.mean(delays) if len(delays) > 0 else 0
    avg_consecutive = np.mean(consecutive_scans_list)
    unique_bands = len(set(actions_taken))
    
    # Selection distribution
    dist = np.bincount(actions_taken, minlength=env.num_bands) / total_scans * 100
    dist_str = ", ".join([f"B{i}:{d:.1f}%" for i, d in enumerate(dist)])
    
    return {
        "hit_rate": hit_rate,
        "interception_rate": interception_rate,
        "missed_opp": missed_opp_rate,
        "avg_delay": avg_delay,
        "avg_reward": avg_reward,
        "avg_consec": avg_consecutive,
        "unique": unique_bands,
        "dist": dist_str
    }

def print_results(name, r):
    print(f"\n{name} Results:")
    print(f"  Hit Rate:                  {r['hit_rate']:.2f}%")
    print(f"  Interception Rate:         {r['interception_rate']:.2f}%")
    print(f"  Missed Opportunity Rate:   {r['missed_opp']:.2f}%")
    print(f"  Avg Interception Delay:    {r['avg_delay']:.2f} steps")
    print(f"  Average Reward per Step:   {r['avg_reward']:.3f}")
    print(f"  Unique Bands Scanned:      {r['unique']} / 10")
    print(f"  Avg Consecutive Scans:     {r['avg_consec']:.2f}")
    print(f"  Band Distribution:         {r['dist']}")

def evaluate_all():
    print("\n==================================================")
    print("PHASE 12: COMPREHENSIVE BASELINE COMPARISON")
    print("==================================================")
    
    test_file = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_6.h5"
    
    env = RFScanEnv(
        filepaths=test_file,
        enable_exploration=False,
        enable_repeat_penalty=False,
        norm_stats_path="environment/norm_generalized.json"
    )
    
    # 1. Random
    rand_agent = RandomScheduler(10)
    rand_res = run_evaluation(env, rand_agent)
    print_results("1. Random Scan", rand_res)
    
    # 2. Round Robin
    rr_agent = RoundRobinScheduler(10)
    rr_res = run_evaluation(env, rr_agent)
    print_results("2. Round Robin", rr_res)
    
    # 3. Adaptive Heuristic
    ah_agent = AdaptiveHeuristicScheduler(10)
    ah_res = run_evaluation(env, ah_agent)
    print_results("3. Adaptive Heuristic", ah_res)
    
    # 4. RL Agent
    rl_agent = PPO.load("models/Model_Generalized_5Files_final.zip")
    rl_res = run_evaluation(env, rl_agent, is_rl=True, stochastic=True)
    print_results("4. RL Smart Scan Scheduler (Stochastic)", rl_res)
    
    print("\n==================================================")
    print("FINAL COMPARISON - IMPROVEMENT OVER BASELINES")
    print("==================================================")
    
    rl_ir = rl_res['interception_rate']
    
    baselines = {
        "Random": rand_res['interception_rate'],
        "Round Robin": rr_res['interception_rate'],
        "Adaptive Heuristic": ah_res['interception_rate']
    }
    
    for name, base_ir in baselines.items():
        absolute = rl_ir - base_ir
        relative = (absolute / base_ir * 100) if base_ir > 0 else float('inf')
        print(f"RL Improvement over {name:<20}: Absolute = +{absolute:>5.2f}% | Relative = +{relative:>6.2f}%")
        
    print("==================================================\n")

if __name__ == "__main__":
    evaluate_all()
