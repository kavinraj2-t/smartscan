# simulator/environment.py
"""
The RF Environment simulator that manages emitters and maintains ground truth.
"""
from typing import List, Dict
from simulator.config import NUM_BANDS
from simulator.emitters import Emitter

class RFEnvironment:
    def __init__(self, emitters: List[Emitter]):
        self.emitters = emitters
        self.current_time = 0

    def step(self) -> Dict[str, any]:
        """
        Advances the simulation by one time step and returns the ground truth.
        """
        # Initialize bands as inactive (0)
        bands_status = [0 for _ in range(NUM_BANDS)]
        active_emitters_info = []

        for emitter in self.emitters:
            active_band = emitter.get_active_band(self.current_time)
            if active_band is not None and 0 <= active_band < NUM_BANDS:
                bands_status[active_band] = 1
                active_emitters_info.append((emitter.name, active_band))

        # Capture state before advancing time
        state = {
            "time": self.current_time,
            "bands": bands_status,
            "active_emitters": active_emitters_info
        }
        
        self.current_time += 1
        return state
