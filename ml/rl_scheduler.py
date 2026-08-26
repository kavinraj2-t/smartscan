# ml/rl_scheduler.py
from stable_baselines3 import PPO
import numpy as np
from simulator.config import NUM_BANDS

class RLScheduler:
    """
    Adapter that allows the trained PPO model to act as a scheduler in the main simulation.
    """
    def __init__(self, model_path="models/ppo_smart_scan.zip"):
        self.model = PPO.load(model_path)
        
        # Maintain local history, identical to what rf_scan_env does
        self.history = []
        for _ in range(NUM_BANDS):
            self.history.append({
                "last_status": 0.0,
                "time_since_scan": 1.0,
                "hit_count": 0.0,
                "miss_count": 0.0
            })
            
    def get_next_band(self, time_step: int) -> int:
        # Get action from the model based on strictly local observations
        obs = self._get_obs()
        action, _states = self.model.predict(obs, deterministic=True)
        return int(action)
        
    def _get_obs(self):
        obs = []
        for h in self.history:
            obs.extend([
                h["last_status"],
                min(h["time_since_scan"] / 50.0, 1.0),
                min(h["hit_count"] / 100.0, 1.0),
                min(h["miss_count"] / 100.0, 1.0)
            ])
        return np.array(obs, dtype=np.float32)

    def update_feedback(self, time_step: int, selected_band: int, result: str):
        # Time passes for all bands
        for i in range(NUM_BANDS):
            self.history[i]["time_since_scan"] += 1.0
            
        self.history[selected_band]["time_since_scan"] = 0.0
        
        # Register the hit/miss
        if result == "HIT":
            self.history[selected_band]["last_status"] = 1.0
            self.history[selected_band]["hit_count"] += 1.0
        else:
            self.history[selected_band]["last_status"] = 0.0
            self.history[selected_band]["miss_count"] += 1.0
