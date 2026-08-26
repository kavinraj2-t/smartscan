# receiver/receiver.py
"""
Receiver model that can scan only ONE band at a time and maintains a scan history.
"""
from typing import Dict, Any

class Receiver:
    def __init__(self):
        self.scan_history = []

    def scan(self, time_step: int, selected_band: int, environment_state: dict) -> Dict[str, Any]:
        """
        Scans a selected band by extracting ONLY the requested band's status
        from the environment state. This ensures the receiver does not have access
        to the hidden ground truth of other bands.
        """
        # The receiver requests observation of ONLY the selected band
        observed_status = environment_state["bands"][selected_band]
        
        # If there's an active transmission, we identify the emitter (simulating signal classification)
        active_emitters = environment_state["active_emitters"]
        emitters_in_band = [name for name, band in active_emitters if band == selected_band]
        
        emitter_id = emitters_in_band[0] if emitters_in_band else None
        
        # Deduce emitter_type from the name format (e.g., "Radar-Continuous" -> "Continuous")
        emitter_type = None
        if emitter_id and "-" in emitter_id:
            emitter_type = emitter_id.split("-")[1]
            
        result = "HIT" if observed_status == 1 else "MISS"
        
        record = {
            "time_step": time_step,
            "selected_band": f"B{selected_band + 1}",
            "observed_status": observed_status,
            "emitter_id": emitter_id,
            "emitter_type": emitter_type,
            "result": result
        }
        self.scan_history.append(record)
        return record
