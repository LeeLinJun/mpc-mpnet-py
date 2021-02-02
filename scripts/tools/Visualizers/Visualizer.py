import numpy as np
from .utils import Dynamics
from abc import ABC


class Visualizer(ABC):
    system_bounds = {
        # , [-6, 6], [-6, 6]],
        Dynamics.Acrobot: np.array([[-np.pi, np.pi], [-np.pi, np.pi]]),
        # , [-40, 40], [-2, 2]],
        Dynamics.Cartpole: np.array([[-30, 30], [-np.pi, np.pi]]),
        Dynamics.Car: np.array([[-25, 25], [-35, 35], [-np.pi, np.pi]]),
        Dynamics.Quadorotor: np.array([[-5, 5], [-5, 5], [-5, 5]])
    }

    def __init__(self, system, figs_axs, draw_v_space=False,
                 start_color='red',
                 goal_color='green',
                 obs_color='black',
                 path_color='blue',
                 ref_path_color='green'):
        assert system is not None
        self.system = system
        self.draw_v_space = draw_v_space
        self.bound = self.system_bounds[self.system]

        self.figs_axs = figs_axs
        self.start_color = start_color
        self.goal_color = goal_color
        self.obs_color = obs_color
        self.path_color = path_color
        self.ref_path_color = ref_path_color

    def plot(self, waypoints, goal, ref_path=None):
        # plot(waypoints, goal, ref_path=None)
        pass
