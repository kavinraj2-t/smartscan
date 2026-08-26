# receiver/random_scan.py
"""
Baseline Random scanning strategy.
"""
import random
from simulator.config import NUM_BANDS, RANDOM_SEED

class RandomScheduler:
    def __init__(self):
        # Use a reproducible random seed for the scheduler
        self.rng = random.Random(RANDOM_SEED + 1) # +1 so it's different from the environment seed
        
    def get_next_band(self, time_step: int) -> int:
        """
        Returns a randomly selected band.
        """
        return self.rng.randint(0, NUM_BANDS - 1)
