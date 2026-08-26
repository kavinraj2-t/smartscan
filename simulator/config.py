# simulator/config.py
"""
Configuration parameters for the Smart Scan simulation environment.
"""

# Define the frequency bands (Index 0 to 4 correspond to B1 to B5)
BANDS = [
    {"id": 1, "name": "B1", "range": (100, 120)},
    {"id": 2, "name": "B2", "range": (120, 140)},
    {"id": 3, "name": "B3", "range": (140, 160)},
    {"id": 4, "name": "B4", "range": (160, 180)},
    {"id": 5, "name": "B5", "range": (180, 200)},
]

NUM_BANDS = len(BANDS)

# Simulation parameters
RANDOM_SEED = 42 # Reproducible random seed
