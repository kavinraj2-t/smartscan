# ml/rf_scan_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from simulator.config import NUM_BANDS
from simulator.emitters import ContinuousEmitter, PeriodicEmitter, AgileEmitter, BurstEmitter
from simulator.environment import RFEnvironment

class RFScanEnv(gym.Env):
    """
    Reinforcement Learning environment wrapping the RF simulator.
    The agent acts as the receiver's scheduler.
    """
    def __init__(self, seed=None):
        super(RFScanEnv, self).__init__()
        
        # Action space: Discrete choice of which band to scan next
        self.action_space = spaces.Discrete(NUM_BANDS)
        
        # Observation space: 4 features per band (last_status, time_since_scan, hit_count, miss_count)
        self.features_per_band = 4
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(NUM_BANDS * self.features_per_band,), 
            dtype=np.float32
        )
        
        self.max_steps = 500
        self._seed = seed
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            
        if self._seed is not None:
            random.seed(self._seed)
            
        # Create a mixed training scenario
        emitters = [
            ContinuousEmitter(name="Radar-Continuous", band_index=0),
            PeriodicEmitter(name="Comm-Periodic", band_index=2, period=5, active_duration=2),
            AgileEmitter(name="Hopping-Agile", band_indices=[1, 3, 4], hop_interval=3),
            BurstEmitter(name="Data-Burst", band_index=1, burst_probability=0.2, max_burst_duration=3)
        ]
        self.env = RFEnvironment(emitters=emitters)
        self.current_step = 0
        
        # Reset observation history
        self.history = []
        for _ in range(NUM_BANDS):
            self.history.append({
                "last_status": 0.0,
                "time_since_scan": 1.0, # 1.0 means hasn't been scanned in a long time (normalized)
                "hit_count": 0.0,
                "miss_count": 0.0
            })
            
        return self._get_obs(), {}
        
    def _get_obs(self):
        """Constructs the observation vector exclusively from previously observed receiver data."""
        obs = []
        for h in self.history:
            # Normalize inputs to [0, 1] range for the neural network
            obs.extend([
                h["last_status"],
                min(h["time_since_scan"] / 50.0, 1.0),
                min(h["hit_count"] / 100.0, 1.0),
                min(h["miss_count"] / 100.0, 1.0)
            ])
        return np.array(obs, dtype=np.float32)
        
    def step(self, action):
        self.current_step += 1
        
        # Advance ground truth (Environment advances, Time + 1)
        state = self.env.step() 
        
        # Increment time since scan for ALL bands in our history
        for i in range(NUM_BANDS):
            self.history[i]["time_since_scan"] += 1.0
            
        # Receiver performs scan on the chosen band
        observed_status = state["bands"][action]
        
        # Update history for the scanned band
        self.history[action]["last_status"] = float(observed_status)
        self.history[action]["time_since_scan"] = 0.0
        
        # Calculate Reward based on hidden state (Allowed for training, but NOT in observation)
        reward = 0.0
        if observed_status == 1:
            reward += 5.0 # Hit
            self.history[action]["hit_count"] += 1.0
        else:
            reward -= 1.0 # Miss
            self.history[action]["miss_count"] += 1.0
            
        # Penalty for missed transmission opportunities
        for i in range(NUM_BANDS):
            if i != action and state["bands"][i] == 1:
                reward -= 3.0 # Missed important signal
                
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}
