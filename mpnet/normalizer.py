import numpy as np

class Normalizer:
    def __init__(self, system='acrobot_obs'):
        self.system = system
        self.modes = {'acrobot_obs': np.array([np.pi, np.pi, 6, 6]),
                     'cartpole_obs': np.array([30, 40, np.pi, 2])}
        self.para = self.modes[self.system]
        
    def normalize(self, state):
        return state / self.para
    
    def denormalize(self, state):
        return state * self.para
    