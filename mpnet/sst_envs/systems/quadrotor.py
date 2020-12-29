import numpy as np


class Quadrotor:
    def __init__(self, svc=None, env=None, verbose=False):
        self.MIN_C1 = -15.
        self.MAX_C1 = 0.
        self.MIN_C = -1.
        self.MAX_C = 1.
        self.MIN_V = -1.
        self.MAX_V = 1.
        self.MIN_W = -1.
        self.MAX_W = 1.
        self.MASS_INV = 1.
        self.BETA = 1.
        self.EPS = 2.107342e-08

    def enforce_bounds_quaternion(self, qstate):
        # enforce quaternion
        # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
        # [x, y, z, w]
        nrmSqr = qstate[0]*qstate[0] + qstate[1]*qstate[1] + \
            qstate[2]*qstate[2] + qstate[3]*qstate[3]
        nrmsq = np.sqrt(nrmSqr) if (np.abs(nrmSqr - 1.0) > 1e-6) else 1.0
        error = np.abs(1.0 - nrmsq)
        if error < self.EPS:
            scale = 2.0 / (1.0 + nrmsq)
            qstate *= scale
        else:
            if nrmsq < 1e-6:
                qstate[:] = 0
                qstate[3] = 1
            else:
                scale = 1.0 / np.sqrt(nrmsq)
                qstate *= scale
        return qstate

    def enforce_bounds_quaternion_vec(self, pose):
        # enforce quaternion
        # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
        # [x, y, z, w]
        nrmsq = np.sum(pose ** 2, axis=1)
        ind = np.abs(1.0 - nrmsq) < self.EPS
        pose[ind, :] *= 2.0 / (1.0 + np.expand_dims(nrmsq[ind], axis=1))
        ind = nrmsq < 1e-6
        pose[ind, 0:3] = 0
        pose[ind, 3] = 1
        pose *= 1.0 / (np.expand_dims(np.sqrt(nrmsq), axis=1) + self.EPS)
        return pose

    def _compute_derivatives(self, q, u):
        qdot = np.zeros(q.shape)
        qdot[0:3] = q[7:10]
        qomega = np.zeros(4)  # [ x, y, z, w,]
        qomega[0:3] = 0.5 * q[10:13]
        qomega = self.enforce_bounds_quaternion(qomega)
        delta = q[3] * qomega[0] + q[4] * qomega[1] + q[5] * qomega[2]
        qdot[3:7] = qomega - delta * q[3:7]
        qdot[7] = self.MASS_INV * \
            (-2 * u[0] * (q[6] * q[4] + q[3] * q[5]) - self.BETA * q[7])
        qdot[8] = self.MASS_INV * \
            (-2 * u[0] * (q[4] * q[5] - q[6] * q[3]) - self.BETA * q[8])
        qdot[9] = self.MASS_INV * (-u[0] * (q[6] * q[6] - q[3] *
                                            q[3] - q[4] * q[4] + q[5] * q[5]) - self.BETA * q[9]) - 9.81
        qdot[10:13] = u[1:4]
        return qdot

    def _compute_derivatives_vec(self, q, u):
        qdot = np.zeros(q.shape)
        qdot[:, 0:3] = q[:, 7:10]
        qomega = np.zeros((q.shape[0], 4))  # [ x, y, z, w,]
        qomega[:, 0:3] = 0.5 * q[:, 10:13]
        qomega = self.enforce_bounds_quaternion_vec(qomega)
        delta = q[:, 3] * qomega[:, 0] + q[:, 4] * \
            qomega[:, 1] + q[:, 5] * qomega[:, 2]
        qdot[:, 3:7] = qomega - np.expand_dims(delta, axis=1) * q[:, 3:7]
        qdot[:, 7] = self.MASS_INV * \
            (-2 * u[:, 0] * (q[:, 6] * q[:, 4] +
                             q[:, 3] * q[:, 5]) - self.BETA * q[:, 7])
        qdot[:, 8] = self.MASS_INV * \
            (-2 * u[:, 0] * (q[:, 4] * q[:, 5] -
                             q[:, 6] * q[:, 3]) - self.BETA * q[:, 8])
        qdot[:, 9] = self.MASS_INV * (-u[:, 0] * (q[:, 6] * q[:, 6] - q[:, 3] *
                                                  q[:, 3] - q[:, 4] * q[:, 4] + q[:, 5] * q[:, 5]) - self.BETA * q[:, 9]) - 9.81
        qdot[:, 10:13] = u[:, 1:4]
        return qdot

    def propagate(self, start_state, control, steps, integration_step):
        '''
        control (n_sample)
        t is (n_sample)
        # control in [NS, NC=4], t in [NS]
        '''
        q = start_state.copy()
        control[0] = np.clip(control[0], self.MIN_C1, self.MAX_C1)
        control[1] = np.clip(control[1], self.MIN_C, self.MAX_C)
        control[2] = np.clip(control[2], self.MIN_C, self.MAX_C)
        control[3] = np.clip(control[3], self.MIN_C, self.MAX_C)
        q[3:7] = self.enforce_bounds_quaternion(q[3:7])
        for t in range(0, steps):
            q += integration_step * self._compute_derivatives(q, control)
            q[7:11] = np.clip(q[7:11], self.MIN_V, self.MAX_V)
            q[10:13] = np.clip(q[10:13], self.MIN_W, self.MAX_W)
            q[3:7] = self.enforce_bounds_quaternion(q[3:7])
        return q

    def propagate_vec(self, start_state, control, t, integration_step, direction=1):
        '''
        control (n_sample)
        t is (n_sample)
        # control in [NS, NC=4], t in [NS]
        '''
        q = start_state
        control[:, 0] = np.clip(control[:, 0], self.MIN_C1, self.MAX_C1)
        control[:, 1] = np.clip(control[:, 1], self.MIN_C, self.MAX_C)
        control[:, 2] = np.clip(control[:, 2], self.MIN_C, self.MAX_C)
        control[:, 3] = np.clip(control[:, 3], self.MIN_C, self.MAX_C)
        q[:, 3:7] = self.enforce_bounds_quaternion_vec(q[:, 3:7])
        t_max = np.max(t)
        for t_curr in np.arange(0, t_max + integration_step, integration_step):
            q[t >= t_curr, :] += direction * integration_step * \
                self._compute_derivatives_vec(
                    q[t >= t_curr, :], control[t >= t_curr, :])
            q[:, 7:11] = np.clip(q[:, 7:11], self.MIN_V, self.MAX_V)
            q[:, 10:13] = np.clip(q[:, 10:13], self.MIN_W, self.MAX_W)
            q[:, 3:7] = self.enforce_bounds_quaternion_vec(q[:, 3:7])
        return q



if __name__ == '__main__':
    quadrotor = QuadrotorVec()

    n_sample = 5
    start_state = np.zeros((n_sample, 13))
    start_state[:, 6] = 1

    start_state[:, 0] = 1
    control = np.ones((n_sample, 4)) * (-5)
    t = np.ones((n_sample))
    print(quadrotor.propagate_vec(start_state, control, t, 2e-2))
