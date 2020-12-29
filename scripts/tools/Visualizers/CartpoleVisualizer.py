# from matplotlib import pyplot as plt
from .Visualizer import Visualizer, Dynamics
from .utils import line_line_cc
import numpy as np
import matplotlib.patches as patches


class CartpoleVisualizer(Visualizer):
    L = 2.5
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2

    dx = 1
    dtheta = 0.1
    width = 4

    pole_l = 2.5
    pole_w = 0.01
    cart_w = 1.
    cart_h = 0.5
    obs_w = 4
    obs_h = 4
    integration_step = 0.002

    def __init__(self, figs_axs, obs=None,
                 start_color='springgreen',
                 goal_color='red',
                 obs_color='slategray',
                 path_color='dodgerblue',
                 ref_path_color='springgreen',
                 cart_color='cornflowerblue'):
        super().__init__(system=Dynamics.Cartpole, fig_axs=figs_axs,
                         start_color=start_color,
                         goal_color=goal_color,
                         obs_color=obs_color,
                         path_color=path_color,
                         ref_path_color=ref_path_color)

        self.obs = obs
        if self.obs is not None:
            self.__draw_obstacles()
        (self.fig_s, self.ax_s), (self.fig_ws, self.ax_ws) = self.figs_axs

    def __draw_path(self, path, color='blue'):
        # draw state trajectory
        self.ax_s.scatter(path[:, 0], path[:, 2], color=color)
        self.ax_s.plot(path[:, 0], path[:, 2], color=color)

        # draw workspace trajectory

    def __in_collision(self, state, obc, obs_width=4.):

        if state[0] < self.MIN_X or state[0] > self.MAX_X:
            return True

        pole_x1 = state[0]
        pole_y1 = self.H
        pole_x2 = state[0] + self.L * np.sin(state[2])
        pole_y2 = self.H + self.L * np.cos(state[2])

        for i in range(len(obc)):
            for j in range(0, 8, 2):
                x1 = obc[i][j]
                y1 = obc[i][j+1]
                x2 = obc[i][(j+2) % 8]
                y2 = obc[i][(j+3) % 8]
                if line_line_cc(pole_x1, pole_y1, pole_x2,
                                pole_y2, x1, y1, x2, y2):
                    return True
        return False

    def __draw_collision_states(self):
        obs_list = []
        for i in range(len(self.obs)):
            x = self.obs[i][0]
            y = self.obs[i][1]
            obs = np.zeros(8)
            obs[0] = x - self.width / 2
            obs[1] = y + self.width / 2

            obs[2] = x + self.width / 2
            obs[3] = y + self.width / 2

            obs[4] = x + self.width / 2
            obs[5] = y - self.width / 2

            obs[6] = x - self.width / 2
            obs[7] = y - self.width / 2
            obs_list.append(obs)
        obs_i = np.array(obs_list)

        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int((self.MAX_X - self.MIN_X) / self.dx)
        jmin = 0
        jmax = int(2 * np.pi / self.dtheta)

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                x = np.array([self.dx * i + self.MIN_X, 0.,
                              self.dtheta * j - np.pi, 0.])
                if self.__in_collision(x, obs_i, obs_width=self.width):
                    infeasible_points.append(x)
                else:
                    feasible_points.append(x)
        feasible_points = np.array(feasible_points)
        infeasible_points = np.array(infeasible_points)
        # self.ax.scatter(feasible_points[:,0],
        #                 feasible_points[:,2], c='azure')
        self.ax_s.scatter(infeasible_points[:, 0],
                          infeasible_points[:, 2],
                          c=self.obs_color)

    def __draw_obstacles(self):
        # draw obs in work space
        # TODO:
        # draw obs in state space
        self.__draw_collision_states(self)

    def __draw_workspace_state(self, state):
        pole = patches.Rectangle((state[0] - self.pole_w/2, self.cart_h),
                                 self.pole_w, self.pole_l,
                                 linewidth=.5, edgecolor=self.path_color,
                                 facecolor=self.path_color)
        cart = patches.Rectangle((state[0]-self.params['cart_w']/2, 0),
                                 self.cart_w, self.cart_h,
                                 linewidth=.5, edgecolor=self.cart_color,
                                 facecolor=self.cart_color)
        self.ax_ws

    def plot(self, waypoints, goal, ref_path=None):

        self.ax_s.scatter(goal[0], goal[2], color=self.goal_color)
        self.__draw_path(waypoints, color=self.path_color)
        if ref_path is not None:
            self.__draw_path(ref_path, color=self.ref_path_color)
