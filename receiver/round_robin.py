# receiver/round_robin.py
"""
Baseline Round-Robin scanning strategy.
"""
from simulator.config import NUM_BANDS

class RoundRobinScheduler:
    def __init__(self):
        self.current_index = 0
        
    def get_next_band(self, time_step: int) -> int:
        """
        Returns the next band in a sequential sequence: B1 -> B2 -> B3 -> B4 -> B5 -> B1...
        """
        band = self.current_index
        self.current_index = (self.current_index + 1) % NUM_BANDS
        return band
