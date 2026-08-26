import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.build_environment import build_environment
from environment.normalization import load_normalization_stats

class RFScanEnv(gym.Env):
    """
    Partially Observable RF Environment.
    The agent only observes the state of the band it explicitly scans.
    """
    
    def __init__(self, filepaths, time_window=50000, num_bands=10, max_consecutive_scans=3, enable_exploration=True, enable_repeat_penalty=True, norm_stats_path="environment/normalization_stats.json"):
        super(RFScanEnv, self).__init__()
        
        if isinstance(filepaths, str):
            self.filepaths = [filepaths]
        else:
            self.filepaths = filepaths
            
        self.time_window = time_window
        self.num_bands = num_bands
        
        # Ablation Flags
        self.enable_exploration = enable_exploration
        self.enable_repeat_penalty = enable_repeat_penalty
        
        # Load Normalization Stats
        self.norm_stats = load_normalization_stats(norm_stats_path)
        self.max_pulse_log = math.log1p(self.norm_stats['max_pulse_count'])
        self.amp_p1 = self.norm_stats['amp_p1']
        self.amp_p99 = self.norm_stats['amp_p99']
        
        # Reward Configuration
        self.HIT_REWARD = 1.0
        self.MISS_PENALTY = -0.05
        self.MAX_EXPLORATION_BONUS = 0.1
        self.MAX_CONSECUTIVE_SCANS = max_consecutive_scans
        self.REPEAT_SCAN_PENALTY_BASE = 0.05
        
        # Gym Spaces
        self.action_space = spaces.Discrete(self.num_bands)
        
        # Observation Space: (NUM_BANDS, 7 features)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_bands, 7), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly select a file from the list for this episode
        selected_file = self.np_random.choice(self.filepaths)
        
        # Load Ground Truth
        self.ground_truth = build_environment(selected_file, time_window=self.time_window, num_bands=self.num_bands)
        self.num_windows = self.ground_truth['num_windows']
        self.max_episode_length = float(self.num_windows)
        
        self.current_time_step = 0
        
        # Receiver's Internal Memory (Unnormalized)
        self.history_last_scan_time = np.full(self.num_bands, -1.0)
        self.history_last_hit_time = np.full(self.num_bands, -1.0)
        self.history_last_pulse_count = np.zeros(self.num_bands)
        self.history_hit_count = np.zeros(self.num_bands)
        self.history_scan_count = np.zeros(self.num_bands)
        self.history_last_mean_amp = np.zeros(self.num_bands)
        
        # Track Consecutive Scans
        self.last_scanned_band = -1
        self.consecutive_scan_count = 0
        
        # Tracking metrics for evaluation
        self.total_reward = 0.0
        self.unique_bands_scanned = set()
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        obs = np.zeros((self.num_bands, 7), dtype=np.float32)
        
        for b in range(self.num_bands):
            # 1. time_since_last_scan
            if self.history_last_scan_time[b] == -1:
                t_scan = self.max_episode_length  # Max uncertainty if never scanned
            else:
                t_scan = self.current_time_step - self.history_last_scan_time[b]
            obs[b, 0] = np.clip(t_scan / self.max_episode_length, 0.0, 1.0)
            
            # 2. time_since_last_hit
            if self.history_last_hit_time[b] == -1:
                t_hit = self.max_episode_length
            else:
                t_hit = self.current_time_step - self.history_last_hit_time[b]
            obs[b, 1] = np.clip(t_hit / self.max_episode_length, 0.0, 1.0)
            
            # 3. last_observed_pulse_count (log normalized)
            obs[b, 2] = np.clip(math.log1p(self.history_last_pulse_count[b]) / self.max_pulse_log, 0.0, 1.0) if self.max_pulse_log > 0 else 0.0
            
            # 4. recent_observed_hit_rate
            if self.history_scan_count[b] > 0:
                obs[b, 3] = self.history_hit_count[b] / self.history_scan_count[b]
            else:
                obs[b, 3] = 0.0
                
            # 5. recent_observed_mean_amplitude
            if self.amp_p99 > self.amp_p1:
                norm_amp = (self.history_last_mean_amp[b] - self.amp_p1) / (self.amp_p99 - self.amp_p1)
                obs[b, 4] = np.clip(norm_amp, 0.0, 1.0)
            else:
                obs[b, 4] = 0.0
                
            # 6. number_of_times_scanned
            obs[b, 5] = np.clip(self.history_scan_count[b] / self.max_episode_length, 0.0, 1.0)
            
            # 7. uncertainty_score (same as time_since_last_scan)
            obs[b, 6] = obs[b, 0]
            
        return obs

    def step(self, action):
        selected_band = int(action)
        self.unique_bands_scanned.add(selected_band)
        
        # Calculate uncertainty BEFORE updating local memory (for exploration bonus)
        if self.history_last_scan_time[selected_band] == -1:
            t_scan = self.max_episode_length
        else:
            t_scan = self.current_time_step - self.history_last_scan_time[selected_band]
        uncertainty = np.clip(t_scan / self.max_episode_length, 0.0, 1.0)
        
        # Update Consecutive Scans
        if selected_band == self.last_scanned_band:
            self.consecutive_scan_count += 1
        else:
            self.last_scanned_band = selected_band
            self.consecutive_scan_count = 1
            
        # Ground Truth checking (Hidden from Observation)
        true_pulse_count = self.ground_truth['pulse_count'][self.current_time_step, selected_band]
        true_mean_amp = self.ground_truth['mean_amplitude'][self.current_time_step, selected_band]
        is_hit = (true_pulse_count > 0)
        
        # Update Receiver Local History
        self.history_last_scan_time[selected_band] = self.current_time_step
        self.history_scan_count[selected_band] += 1
        self.history_last_pulse_count[selected_band] = true_pulse_count
        self.history_last_mean_amp[selected_band] = true_mean_amp
        
        if is_hit:
            self.history_last_hit_time[selected_band] = self.current_time_step
            self.history_hit_count[selected_band] += 1
            
        # ----------------------------------------
        # REWARD CALCULATION
        # ----------------------------------------
        hit_reward = 0.0
        activity_reward = 0.0
        miss_penalty = 0.0
        exploration_bonus = 0.0
        repeat_scan_penalty = 0.0
        
        if is_hit:
            hit_reward = self.HIT_REWARD
            activity_reward = np.clip(math.log1p(true_pulse_count) / self.max_pulse_log, 0.0, 1.0) if self.max_pulse_log > 0 else 0.0
        else:
            miss_penalty = self.MISS_PENALTY
            
        # Exploration Bonus (Bounded)
        if self.enable_exploration:
            exploration_bonus = self.MAX_EXPLORATION_BONUS * uncertainty
        
        # Repeated Scan Penalty
        if self.enable_repeat_penalty and (self.consecutive_scan_count > self.MAX_CONSECUTIVE_SCANS):
            excess = self.consecutive_scan_count - self.MAX_CONSECUTIVE_SCANS
            repeat_scan_penalty = - (self.REPEAT_SCAN_PENALTY_BASE * excess)
            
        reward = hit_reward + activity_reward + miss_penalty + exploration_bonus + repeat_scan_penalty
        self.total_reward += reward
        
        # Advance Time
        self.current_time_step += 1
        terminated = (self.current_time_step >= self.num_windows)
        truncated = False
        
        obs = self._get_observation()
        
        info = {
            "is_hit": is_hit,
            "hit_reward": hit_reward,
            "activity_reward": activity_reward,
            "miss_penalty": miss_penalty,
            "exploration_bonus": exploration_bonus,
            "repeat_scan_penalty": repeat_scan_penalty,
            "consecutive_scans": self.consecutive_scan_count,
            "unique_bands_explored": len(self.unique_bands_scanned)
        }
        
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Test script to validate environment shape and normalization bounds
    filepath = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan/config_0.h5"
    env = RFScanEnv(filepath)
    obs, info = env.reset()
    
    print("\n--- Testing Gymnasium Environment ---")
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape == (10, 7)
    assert np.all((obs >= 0.0) & (obs <= 1.0)), "Normalization failure in reset"
    
    print("Testing 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1} | Action: Band {action} | Reward: {reward:.3f}")
        print(f"Info: {info}")
        assert np.all((obs >= 0.0) & (obs <= 1.0)), f"Normalization failure at step {i+1}"
        
    print("\nEnvironment validation successful! Observation space is strictly [0.0, 1.0].")
