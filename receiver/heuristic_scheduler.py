# receiver/heuristic_scheduler.py
"""
Adaptive Heuristic Scheduler based on rules and past observations.
"""
import random
import math
from simulator.config import NUM_BANDS, RANDOM_SEED

class AdaptiveHeuristicScheduler:
    def __init__(self, epsilon=0.20):
        self.epsilon = epsilon
        # Use a different reproducible seed
        self.rng = random.Random(RANDOM_SEED + 99)
        
        # Maintain history for each band
        self.bands_info = []
        for i in range(NUM_BANDS):
            self.bands_info.append({
                "band_index": i,
                "hit_count": 0,
                "miss_count": 0,
                "total_observations": 0,
                "last_seen_time": -1,
                "last_scanned_time": -1,
                "detection_history": []
            })
        self.last_decision_explanation = ""

    def get_next_band(self, time_step: int) -> int:
        """
        Selects the next band using an epsilon-greedy approach based on priority score.
        """
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            selected_band = self.rng.randint(0, NUM_BANDS - 1)
            self.last_decision_explanation = f"Random exploration (epsilon={self.epsilon})"
            return selected_band
            
        best_band = 0
        best_score = -float('inf')
        best_explanation = ""

        # Calculate score for each band
        for b_info in self.bands_info:
            score, explanation = self._calculate_priority(b_info, time_step)
            if score > best_score:
                best_score = score
                best_band = b_info["band_index"]
                best_explanation = explanation

        self.last_decision_explanation = best_explanation
        return best_band
        
    def _calculate_priority(self, b_info, current_time):
        """
        Calculates the priority score based on Activity, Recency, Periodicity, and Exploration.
        """
        # 1. Activity Score (hit rate of this specific band)
        obs = b_info["total_observations"]
        activity_score = b_info["hit_count"] / obs if obs > 0 else 0.0
        
        # 2. Recency Score (decaying priority since last seen)
        recency_score = 0.0
        if b_info["last_seen_time"] != -1:
            time_since_last_seen = current_time - b_info["last_seen_time"]
            recency_score = math.exp(-0.1 * time_since_last_seen)
            
        # 3. Periodicity Score (predicting periodic transmissions)
        periodicity_score = 0.0
        period_str = "None"
        history = b_info["detection_history"]
        if len(history) >= 2:
            intervals = [history[i] - history[i-1] for i in range(1, len(history))]
            avg_interval = sum(intervals) / len(intervals)
            
            time_since_last = current_time - b_info["last_seen_time"]
            diff = abs(time_since_last - avg_interval)
            
            # Boost score if we are near the expected interval
            if diff <= 1:
                periodicity_score = 2.0
            elif diff <= 2:
                periodicity_score = 1.0
            period_str = f"~{avg_interval:.1f}"

        # 4. Exploration Score (penalty for starvation)
        exploration_score = 0.0
        if b_info["last_scanned_time"] != -1:
            time_since_last_scan = current_time - b_info["last_scanned_time"]
            exploration_score = 0.05 * time_since_last_scan
        else:
            exploration_score = 5.0 # High priority if never scanned
            
        total_score = activity_score + recency_score + periodicity_score + exploration_score
        
        # Build human-readable explanation
        exp = []
        if activity_score > 0: exp.append(f"{b_info['hit_count']} hits")
        if recency_score > 0.5: exp.append("Recent activity")
        if periodicity_score > 0: exp.append(f"Expected periodic (interval {period_str})")
        if exploration_score >= 1.0: exp.append(f"Not scanned in {current_time - b_info['last_scanned_time']} steps")
        
        reason = ", ".join(exp) if exp else "Baseline score"
        explanation = f"Score: {total_score:.2f} | Reasons: {reason}"
        return total_score, explanation
        
    def update_feedback(self, time_step: int, selected_band: int, result: str):
        """
        Updates the scheduler's internal knowledge base AFTER a scan is performed.
        """
        b_info = self.bands_info[selected_band]
        b_info["total_observations"] += 1
        b_info["last_scanned_time"] = time_step
        
        if result == "HIT":
            b_info["hit_count"] += 1
            b_info["last_seen_time"] = time_step
            b_info["detection_history"].append(time_step)
        else:
            b_info["miss_count"] += 1

