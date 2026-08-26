class RoundRobinScheduler:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        self.current_band = 0
        
    def get_action(self, observation):
        action = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return action
