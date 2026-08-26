# metrics/metrics.py
"""
Metrics calculation for evaluating the performance of scanning schedulers.
"""
from typing import List, Dict, Any

class MetricsCalculator:
    def __init__(self, ground_truth_history: List[Dict[str, Any]], scan_history: List[Dict[str, Any]]):
        self.ground_truth = ground_truth_history
        self.scan_history = scan_history

    def calculate_metrics(self) -> Dict[str, float]:
        total_scans = len(self.scan_history)
        total_hits = sum(1 for scan in self.scan_history if scan["result"] == "HIT")
        total_misses = total_scans - total_hits
        
        # 1. Reconstruct transmission episodes to calculate delay
        # A transmission episode is a contiguous block of time an emitter is active.
        episodes = []
        current_active = {} # maps emitter_name to its current episode dict
        
        for state in self.ground_truth:
            t = state["time"]
            active_emitters_now = {name for name, band in state["active_emitters"]}
            
            # Start new episodes or continue existing ones
            for name in active_emitters_now:
                if name not in current_active:
                    episode = {"emitter": name, "start": t, "end": t, "detected_at": None}
                    current_active[name] = episode
                    episodes.append(episode)
                else:
                    current_active[name]["end"] = t
            
            # End episodes for emitters that stopped transmitting
            ended_emitters = [name for name in current_active if name not in active_emitters_now]
            for name in ended_emitters:
                del current_active[name]
                
        # 2. Count missed transmissions and update detected_at for episodes
        total_transmission_opportunities = 0
        missed_transmissions = 0
        
        for gt, scan in zip(self.ground_truth, self.scan_history):
            scanned_band_str = scan["selected_band"] # e.g. "B3"
            scanned_band_idx = int(scanned_band_str[1:]) - 1
            
            total_transmission_opportunities += len(gt["active_emitters"])
            
            for name, band_idx in gt["active_emitters"]:
                if band_idx != scanned_band_idx:
                    missed_transmissions += 1
                    
                # Check if this scan detected an active episode
                for ep in episodes:
                    if ep["emitter"] == name and ep["start"] <= gt["time"] <= ep["end"]:
                        # If the receiver scanned the correct band and hasn't detected it yet
                        if band_idx == scanned_band_idx and ep["detected_at"] is None:
                            ep["detected_at"] = gt["time"]

        # Calculate Delay
        total_detected_episodes = sum(1 for ep in episodes if ep["detected_at"] is not None)
        total_delay = sum(ep["detected_at"] - ep["start"] for ep in episodes if ep["detected_at"] is not None)
        
        average_intercept_delay = total_delay / total_detected_episodes if total_detected_episodes > 0 else 0.0
        
        # Calculate Rates
        interception_rate = (total_hits / total_transmission_opportunities) * 100 if total_transmission_opportunities > 0 else 0.0
        hit_rate = (total_hits / total_scans) * 100 if total_scans > 0 else 0.0
        percentage_of_missed = (missed_transmissions / total_transmission_opportunities) * 100 if total_transmission_opportunities > 0 else 0.0

        return {
            "total_scans": total_scans,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "interception_rate": interception_rate,
            "average_intercept_delay": average_intercept_delay,
            "missed_transmissions": missed_transmissions,
            "percentage_of_missed_transmissions": percentage_of_missed
        }
