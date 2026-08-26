import random

class RandomScheduler:
    def __init__(self, num_bands, seed=42):
        self.num_bands = num_bands
        random.seed(seed)
        
    def get_action(self, observation):
        return random.randint(0, self.num_bands - 1)
