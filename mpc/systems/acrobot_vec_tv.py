import numpy as np

class Acrobot:
    '''
    Two joints pendulum that is activated in the second joint (Acrobot)
    '''
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81
    obs_wh = 6

    def propagate(self, start_state, control, num_steps, integration_step):
        state = start_state
        num_steps_max = np.max(num_steps)
        for i in range((num_steps_max).astype(np.int)):
            active_mask = num_steps > i
            state[active_mask, :] += integration_step * self._compute_derivatives(state[active_mask, :], control[active_mask, :])
            state[state[:, 0] < -np.pi, 0] += 2*np.pi
            state[state[:, 0] > np.pi, 0] -= 2*np.pi
            state[state[:, 1] < -np.pi, 1] += 2*np.pi
            state[state[:, 1] > np.pi, 1] -= 2*np.pi
            state[:, 2:] = np.clip(
                state[:, 2:],
                [self.MIN_V_1, self.MIN_V_2],
                [self.MAX_V_1, self.MAX_V_2])
        return state

    def _compute_derivatives(self, state, control):
        '''
        Port of the cpp implementation for computing state space derivatives
        '''
        theta2 = state[:, self.STATE_THETA_2]
        theta1 = state[:, self.STATE_THETA_1] - np.pi/2
        theta1dot = state[:, self.STATE_V_1]
        theta2dot = state[:, self.STATE_V_2]
        _tau = np.clip(
                control[:, 0],
                self.MIN_TORQUE,
                self.MAX_TORQUE)
#         _tau = control[:, 0]
        m = self.m
        l2 = self.l2
        lc2 = self.lc2
        l = self.l
        lc = self.lc
        I1 = self.I1
        I2 = self.I2

        d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * np.cos(theta2)) + I1 + I2
        d22 = m * lc2 + I2
        d12 = m * (lc2 + l * lc * np.cos(theta2)) + I2
        d21 = d12

        c1 = -m * l * lc * theta2dot * theta2dot * np.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * np.sin(theta2))
        c2 = m * l * lc * theta1dot * theta1dot * np.sin(theta2)
        g1 = (m * lc + m * l) * self.g * np.cos(theta1) + (m * lc * self.g * np.cos(theta1 + theta2))
        g2 = m * lc * self.g * np.cos(theta1 + theta2)

        deriv = state.copy()
        deriv[:, self.STATE_THETA_1] = theta1dot
        deriv[:, self.STATE_THETA_2] = theta2dot

        u2 = _tau - 1 * .1 * theta2dot
        u1 = -1 * .1 * theta1dot
        theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
        theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
        deriv[:, self.STATE_V_1] = theta1dot_dot
        deriv[:, self.STATE_V_2] = theta2dot_dot
        return deriv

    def get_state_bounds(self):
        return [(self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_V_1, self.MAX_V_1),
                (self.MIN_V_2, self.MAX_V_2)]

    def get_control_bounds(self):
        return [(self.MIN_TORQUE, self.MAX_TORQUE)]
    
    def valid_state(self, temp_state, obs_list):
        if obs_list is None:
            return True
        pole_x0 = np.zeros(temp_state.shape[0])
        pole_y0 = np.zeros(temp_state.shape[0])
        pole_x1 = (self.LENGTH) * np.cos(temp_state[:, self.STATE_THETA_1] - np.pi / 2)
        pole_y1 = (self.LENGTH) * np.sin(temp_state[:, self.STATE_THETA_1] - np.pi / 2)
        pole_x2 = pole_x1 + (self.LENGTH) * np.cos(temp_state[:, self.STATE_THETA_1] + temp_state[:, self.STATE_THETA_2] - np.pi / 2)
        pole_y2 = pole_y1 + (self.LENGTH) * np.sin(temp_state[:, self.STATE_THETA_1] + temp_state[:, self.STATE_THETA_2] - np.pi / 2)
        intersect = np.ones(temp_state.shape[0]).astype(np.bool)
        for i in range(len(obs_list)):
            # check each line of the obstacle
            x1 = obs_list[i][0] - self.obs_wh/2
            y1 = obs_list[i][1] - self.obs_wh/2
            x2 = obs_list[i][0] + self.obs_wh/2
            y2 = obs_list[i][1] + self.obs_wh/2
            intersect = np.logical_and(intersect, np.logical_not(self.lineLine(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2)))
            intersect = np.logical_and(intersect, np.logical_not(self.lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2)))
        return intersect

    def lineLine(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        x1~y4 all in dim (N), which are 1-dim 
        """
        intersect = np.zeros(x1.shape[0]).astype(np.bool)
        uA = np.array(((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)))
        uB = np.array(((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)))
        # if uA and uB are between 0-1, lines are colliding
        #print(np.logical_and(np.logical_and(np.logical_and(uA >= 0, uA <= 1), uB >= 0), uB <= 1))
        intersect[np.logical_and(np.logical_and(np.logical_and(uA >= 0, uA <= 1), uB >= 0), uB <= 1)] = True
        return intersect

    def angular_error(self, state, goal):
        error = np.abs(state - goal)
        error[error > np.pi] = np.pi * 2 - error[error > np.pi]
        return error
    
    def get_loss(self, state, goal, weight):
#         return self.get_distance(state, goal, weight)
        return self.angular_error(state[:, self.STATE_THETA_1], goal[self.STATE_THETA_1]) * weight[self.STATE_THETA_1] +\
    self.angular_error(state[:, self.STATE_THETA_2], goal[self.STATE_THETA_2])* weight[self.STATE_THETA_2] +\
    (state[:, self.STATE_V_1] - goal[self.STATE_V_1]) ** 2 * weight[self.STATE_V_1] +\
    (state[:, self.STATE_V_2] - goal[self.STATE_V_2]) ** 2 * weight[self.STATE_V_2]
    
    def get_distance(self, state, goal, weight):
        LENGTH = 20.
        x = LENGTH*(np.cos(state[:, 0] - np.pi / 2)+np.cos(state[:, 0] + state[:, 1] - np.pi / 2))
        y = LENGTH*(np.sin(state[:, 0] - np.pi / 2)+np.sin(state[:, 0] + state[:, 1] - np.pi / 2))
        x2 = LENGTH*(np.cos(goal[0] - np.pi / 2)+np.cos(goal[0] + goal[1] - np.pi / 2))
        y2 = LENGTH*(np.sin(goal[0] - np.pi / 2)+np.sin(goal[0] + goal[1] - np.pi / 2))
        return np.sqrt((x-x2)**2+(y-y2)**2)
#         return np.sqrt(np.sum((state - goal)[:, :2] **2, axis=1))
#         return np.max(np.abs(state-goal), axis=1)