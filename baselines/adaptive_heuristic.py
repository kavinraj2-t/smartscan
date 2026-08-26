import numpy as np

class AdaptiveHeuristicScheduler:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        
        # Priority weights based on Phase 5 definition
        self.W_HIT_RATE = 2.0
        self.W_UNCERTAINTY = 1.0
        self.W_ACTIVITY = 1.5
        
    def get_action(self, observation):
        """
        observation shape is (num_bands, 7)
        Features: 
        0: time_since_last_scan
        1: time_since_last_hit
        2: last_observed_pulse_count
        3: recent_observed_hit_rate
        4: recent_observed_mean_amplitude
        5: number_of_times_scanned
        6: uncertainty_score
        """
        priorities = np.zeros(self.num_bands)
        
        for b in range(self.num_bands):
            hit_rate = observation[b, 3]
            uncertainty = observation[b, 6]
            activity = observation[b, 2]
            
            priority = (self.W_HIT_RATE * hit_rate) + \
                       (self.W_UNCERTAINTY * uncertainty) + \
                       (self.W_ACTIVITY * activity)
            
            priorities[b] = priority
            
        # Select the band with the highest priority
        # If there's a tie (e.g. at start when all are 0), argmax returns the first, 
        # so we can add a tiny random noise to break ties if we want, or just let it scan sequentially.
        # Let's add tiny noise for tie-breaking
        noise = np.random.uniform(0, 0.0001, size=self.num_bands)
        priorities += noise
        
        return int(np.argmax(priorities))
