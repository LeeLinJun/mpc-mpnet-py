import numpy as np

class CEM:
    def __init__(self, dynamics, params, verbose=False):
        self.update_setting(params)
        self.dynamics = dynamics
        self.params = params
        self.verbose = verbose

    def update_setting(self, params):
        self.n_sample = params['n_sample']
        self.n_elite = params['n_elite']
        self.n_t = params['n_t']

        self.weights = params['weights']
        
        self.mu_u = params['mu_u']
        self.sigma_u = params['sigma_u']
        
        self.mu_u = params['mu_u']
        self.sigma_u = params['sigma_u']
        
        self.mu_t = params['mu_t']
        self.sigma_t = params['sigma_t']
        self.t_min = params['t_min']
        self.t_max = params['t_max']
        
        self.t = params['t']
        self.dt = params['dt']
        self.state_dim = params['state_dim']
        self.control_dim = params['control_dim']

    def solve(self, start, goal, direction=1, obs_list=None):
        """[summary]
        
        Arguments:
            start {[float]} -- np.array in state_dim
            goal {[float]} -- np.array in state_dim
        
        Keyword Arguments:
            direction {int} -- +1 or -1, direction of dynamics (default: {1})
        
        Returns:
            [best_u, best_t] -- best controls in n_t*control_dim and n_t
        """

        ## initialization
        min_loss = np.inf
        start_expand = np.repeat(np.expand_dims(start,axis=0), self.n_sample, axis=0)
        early_stop_count = 0
        ## initialization ends

        ## iteration loop
        for _ in range(200):
            ## iteration initialization
            # u vec in dim(n_t*n_control_dim) -> u_sample in n_sample * (n_t*control_dim)
            t = np.random.multivariate_normal(self.mu_t, self.sigma_t, self.n_sample)
            t = np.clip(t, self.t_min, self.t_max)
            
            u_sample = np.random.multivariate_normal(self.mu_u, self.sigma_u, self.n_sample)
            u_sample = np.clip(u_sample, self.dynamics.MIN_TORQUE, self.dynamics.MAX_TORQUE)
            u = u_sample.reshape(self.n_sample, self.n_t, self.control_dim)
            x = start_expand.copy()
            loss = np.zeros(self.n_sample)
            active_ind = np.ones(self.n_sample).astype(np.bool)

            ## iteration initialization ends
            for j in range(self.n_t):
                if active_ind.sum() > 0:
                    x[active_ind] = self.dynamics.propagate(x[active_ind], u[active_ind, j, :], (t[active_ind, j]/self.params['dt']).astype(int), self.params['dt'])
                    #x = self.dynamics.propagate(x, u[:, j, :], int(self.t/self.params['dt']), self.params['dt'])

                    loss[active_ind] += t[active_ind, j]
#                     loss[active_ind] += self.dynamics.get_loss(x[active_ind], goal, self.weights)
                    ##  state validate        
                    if obs_list is not None:
                        # check collision for every state
                        collision_idx = np.logical_not(self.dynamics.valid_state(x, obs_list)) # boolean array for not valid
                        loss[collision_idx] += 100
    #                 if j >= 2:
                    active_ind = np.logical_and(active_ind, self.dynamics.get_loss(x, goal, self.weights) > self.params['converge_radius'])

                ##  state validate ends

            ## update statistics
            terminal_loss = self.dynamics.get_loss(x, goal, self.weights)
            loss += terminal_loss
            rank = np.argsort(loss)[:self.n_elite]
            if self.verbose:
                print(loss[rank[0]], min_loss)

            if loss[rank[0]] < min_loss:
                min_loss = loss[rank[0]]
                best_u = u[rank[0], :, :]
                best_t = t[rank[0], :]
                self.mu_u = np.mean(u_sample[rank,:], axis=0)
                self.sigma_u = np.cov(u_sample[rank,:].T)

                self.mu_t = np.mean(t[rank], axis=0)
                self.sigma_t = np.cov(t[rank,:].T)
            ## update statistics end

            ## early stop
            if loss[rank[0]] >= min_loss - 1e-2:
                early_stop_count += 1
                self.sigma_u *= 1 + 0.1
                self.sigma_t *= 1 + 0.1
            if min_loss < self.params['converge_radius'] or early_stop_count > self.params['max_it'] or terminal_loss[rank[0]] < self.params['converge_radius']: 
                break
            if loss[rank[0]] > 1:
                # self.update_setting(self.params)
                if loss[rank[0]] > 1e2:
                    break
            ## early stop ends

        ## iteration loop ends

        return best_u, best_t
