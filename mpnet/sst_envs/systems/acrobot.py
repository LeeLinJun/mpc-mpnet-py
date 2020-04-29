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

    def propagate(self, start_state, control, num_steps, integration_step):
        state = start_state
        for i in range(num_steps):
            state += integration_step * self._compute_derivatives(state, control)

            if state[0] < -np.pi:
                state[0] += 2*np.pi
            elif state[0] > np.pi:
                state[0] -= 2 * np.pi
            if state[1] < -np.pi:
                state[1] += 2*np.pi
            elif state[1] > np.pi:
                state[1] -= 2 * np.pi

            state[2:] = np.clip(
                state[2:],
                [self.MIN_V_1, self.MIN_V_2],
                [self.MAX_V_1, self.MAX_V_2])
        return state

    def visualize_point(self, state):
        x = self.LENGTH * np.cos(state[self.STATE_THETA_1] - np.pi / 2) + self.LENGTH * np.cos(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi/2)
        y = self.LENGTH * np.sin(state[self.STATE_THETA_1] - np.pi / 2) + self.LENGTH * np.sin(state[self.STATE_THETA_1] + state[self.STATE_THETA_2] - np.pi / 2)
        x = (x + 2 * self.LENGTH) / (4 * self.LENGTH)
        y = (y + 2 * self.LENGTH) / (4 * self.LENGTH)
        return x, y

    def _compute_derivatives(self, state, control):
        '''
        Port of the cpp implementation for computing state space derivatives
        '''
        theta2 = state[self.STATE_THETA_2]
        theta1 = state[self.STATE_THETA_1] - np.pi/2
        theta1dot = state[self.STATE_V_1]
        theta2dot = state[self.STATE_V_2]
        _tau = np.clip(
            control[0],
            self.MIN_TORQUE,
            self.MAX_TORQUE)
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
        deriv[self.STATE_THETA_1] = theta1dot
        deriv[self.STATE_THETA_2] = theta2dot

        u2 = _tau - 1 * .1 * theta2dot
        u1 = -1 * .1 * theta1dot
        theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
        theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
        deriv[self.STATE_V_1] = theta1dot_dot
        deriv[self.STATE_V_2] = theta2dot_dot
        return deriv

    def get_state_bounds(self):
        return [(self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_ANGLE, self.MAX_ANGLE),
                (self.MIN_V_1, self.MAX_V_1),
                (self.MIN_V_2, self.MAX_V_2)]

    def get_control_bounds(self):
        return [(self.MIN_TORQUE, self.MAX_TORQUE)]

