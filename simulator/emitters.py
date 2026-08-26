# simulator/emitters.py
"""
Definitions for different types of RF emitters.
"""
import random
from typing import Optional, List

class Emitter:
    """Base class for an RF emitter."""
    def __init__(self, name: str):
        self.name = name

    def get_active_band(self, time_step: int) -> Optional[int]:
        """
        Returns the band index (0 to 4) where the emitter is active at the given time step,
        or None if the emitter is not transmitting.
        """
        raise NotImplementedError


class ContinuousEmitter(Emitter):
    """An emitter that constantly transmits in a single fixed band."""
    def __init__(self, name: str, band_index: int):
        super().__init__(name)
        self.band_index = band_index

    def get_active_band(self, time_step: int) -> Optional[int]:
        return self.band_index


class PeriodicEmitter(Emitter):
    """An emitter that transmits at fixed intervals in a fixed band."""
    def __init__(self, name: str, band_index: int, period: int, active_duration: int):
        super().__init__(name)
        self.band_index = band_index
        self.period = period
        self.active_duration = active_duration

    def get_active_band(self, time_step: int) -> Optional[int]:
        # Active if time_step % period < active_duration
        if (time_step % self.period) < self.active_duration:
            return self.band_index
        return None


class AgileEmitter(Emitter):
    """An emitter that hops between a predefined set of bands."""
    def __init__(self, name: str, band_indices: List[int], hop_interval: int):
        super().__init__(name)
        self.band_indices = band_indices
        self.hop_interval = hop_interval

    def get_active_band(self, time_step: int) -> Optional[int]:
        # Determine current band based on the hop interval
        hop_index = (time_step // self.hop_interval) % len(self.band_indices)
        return self.band_indices[hop_index]


class BurstEmitter(Emitter):
    """An emitter that transmits randomly for short bursts."""
    def __init__(self, name: str, band_index: int, burst_probability: float, max_burst_duration: int):
        super().__init__(name)
        self.band_index = band_index
        self.burst_probability = burst_probability
        self.max_burst_duration = max_burst_duration
        
        self.current_burst_remaining = 0

    def get_active_band(self, time_step: int) -> Optional[int]:
        if self.current_burst_remaining > 0:
            self.current_burst_remaining -= 1
            return self.band_index
        else:
            # Check if a new burst starts
            if random.random() < self.burst_probability:
                self.current_burst_remaining = random.randint(1, self.max_burst_duration)
                return self.band_index
        return None
