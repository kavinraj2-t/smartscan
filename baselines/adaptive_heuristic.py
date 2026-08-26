import numpy as np

class AdaptiveHeuristicScheduler:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        
        # Manually tuned baseline weights
        self.HEURISTIC_WEIGHTS = {
            "activity": 0.35,
            "hit_rate": 0.20,
            "uncertainty": 0.20,
            "hit_recency": 0.15,
            "amplitude": 0.10
        }
        
        self.NEUTRAL_HIT_RECENCY = 0.5
        self.REPEAT_SCAN_DECAY = 0.95
        
        self.reset()
        
    def reset(self):
        self.consecutive_scans = 0
        self.last_action = -1
        self.last_priorities = np.zeros(self.num_bands)

    def get_priority_scores(self):
        return self.last_priorities

    def select_band(self, observation):
        """
        observation shape is (num_bands, 7)
        Features from RFScanEnv:
        0: time_since_last_scan
        1: time_since_last_hit
        2: last_observed_pulse_count
        3: recent_observed_hit_rate
        4: recent_observed_mean_amplitude
        5: number_of_times_scanned
        6: uncertainty_score
        """
        priorities = np.zeros(self.num_bands)
        uncertainties = np.zeros(self.num_bands)
        
        for b in range(self.num_bands):
            time_since_scan = observation[b, 0]
            time_since_hit = observation[b, 1]
            last_pulse_count = observation[b, 2]
            hit_rate = observation[b, 3]
            mean_amp = observation[b, 4]
            num_scans = observation[b, 5]
            uncertainty = observation[b, 6]
            
            uncertainties[b] = uncertainty
            
            # Hit Recency Logic
            if num_scans == 0.0 or hit_rate == 0.0:
                hit_recency_score = self.NEUTRAL_HIT_RECENCY
            else:
                hit_recency_score = 1.0 - time_since_hit
                
            priority = (self.HEURISTIC_WEIGHTS["activity"] * last_pulse_count) + \
                       (self.HEURISTIC_WEIGHTS["hit_rate"] * hit_rate) + \
                       (self.HEURISTIC_WEIGHTS["uncertainty"] * uncertainty) + \
                       (self.HEURISTIC_WEIGHTS["hit_recency"] * hit_recency_score) + \
                       (self.HEURISTIC_WEIGHTS["amplitude"] * mean_amp)
                       
            # Repeat Scan Penalty
            if b == self.last_action and self.consecutive_scans > 3:
                priority *= self.REPEAT_SCAN_DECAY
                
            priorities[b] = priority
            
        self.last_priorities = priorities
        
        # Tie-Breaking Logic
        max_priority = np.max(priorities)
        
        # Find all bands that are within a tiny epsilon of the max priority
        epsilon = 1e-6
        tied_bands = np.where(np.abs(priorities - max_priority) < epsilon)[0]
        
        if len(tied_bands) > 1:
            # 1. Prefer higher uncertainty among ties
            tied_uncertainties = uncertainties[tied_bands]
            max_uncertainty = np.max(tied_uncertainties)
            uncertain_ties = tied_bands[np.abs(tied_uncertainties - max_uncertainty) < epsilon]
            
            # 2. Random fallback
            if len(uncertain_ties) > 1:
                selected_band = int(np.random.choice(uncertain_ties))
            else:
                selected_band = int(uncertain_ties[0])
        else:
            selected_band = int(tied_bands[0])
            
        return selected_band

    def update(self, action, observation=None):
        # The environment already updates the observation history.
        # We just need to track consecutive scans for our decay logic.
        if action == self.last_action:
            self.consecutive_scans += 1
        else:
            self.last_action = action
            self.consecutive_scans = 1
            
    # Alias for compatibility with simple baselines (like Random/RoundRobin)
    def get_action(self, observation):
        action = self.select_band(observation)
        self.update(action)
        return action
