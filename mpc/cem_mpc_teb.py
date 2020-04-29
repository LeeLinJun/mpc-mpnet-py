import numpy as np
from .cem_teb import CEM

class MPC:
    def __init__(self, params, model, verbose=True):
        self.model = model
        self.params = params
        self.verbose = verbose
        self.cem = CEM(model, self.params, verbose=self.verbose)

    def solve(self, start, goal, obs_list=None):
        path = []
        self.cem.update_setting(self.params)
        best_u, best_t = self.cem.solve(start.copy(), goal.copy(), obs_list=obs_list)
        x = np.expand_dims(start.copy(), axis=0)
        path.append(list(x[0]))
        min_loss = np.inf
        collision = False
        best_x = x[0].copy()
        bestit = 0

        for ti in range(self.params['n_t']):
            x = self.model.propagate(x.copy(),
                np.expand_dims(best_u[ti], axis=0),
                (best_t[ti]*np.ones(1)/self.params['dt']).astype(int),
                self.params['dt']).copy()
            if obs_list is not None:
                if not self.model.valid_state(x, obs_list)[0]:
                    collision = True
                    if self.verbose:
                        print("collision at {},{}".format(ti, x[0]))
                    break
            path.append(list(x[0]))
            loss = self.model.get_loss(x, goal, self.params['weights'])
            if loss < min_loss:
                min_loss = loss
                best_x = x[0].copy()
                bestit = ti+1
                if min_loss < self.params['converge_radius']:
                    if self.verbose:
                        print('converged at ', ti)
                    break
            if self.model.get_distance(x, self.params['final_goal'], self.params['weights'])< self.params['goal_radius']:
#             if self.model.get_loss(x, self.params['final_goal'], self.params['weights'])< self.params['goal_radius']:

                best_x = x[0].copy()
                bestit = ti+1
                min_loss = - np.inf
                break
        if self.verbose:
            print('loss:',min_loss)
        return best_x.copy(), min_loss, path[:bestit+1], collision, best_t[:bestit+1], best_u[:bestit+1]
    

    def rolling(self, start, goal, obs_list=None):
        self.cem.update_setting(self.params)
        path = []
        x = np.expand_dims(start.copy(), axis=0)
        loss = np.inf
        best_it = 0
        it = 0
        count = 0
        min_loss = np.inf
        best_x = x[0].copy()
        collision = False
        bestit = 0
        costs = []

        for _ in range(self.params['rolling_count']):  
            self.cem.update_setting(self.params)
            best_u, best_t = self.cem.solve(x[0].copy(), goal.copy(), obs_list=obs_list)
            
            x = self.model.propagate(x.copy(),
                np.expand_dims(best_u[0], axis=0),
                (best_t[0]*np.ones(1)/self.params['dt']).astype(int),
                self.params['dt']).copy()
            costs.append(best_t[0])
            if obs_list is not None:
                if not self.model.valid_state(x, obs_list)[0]:
                    collision = True
                    print("collision at {},{}".format(ti, x[0]))
                    break
            path.append(list(x[0]))
            it += 1

            loss = self.model.get_loss(x, goal, self.params['weights'])[0]
            if self.verbose:
                print('loss',loss, 'count', count, 'u:',best_u[0])
            if loss < min_loss:
                count = 0
                best_it = it +1
                
                min_loss = loss
                print('min_loss:', min_loss)
                best_x = x[0].copy()
                if min_loss < self.params['converge_radius']:
                    print('converged')
                    break
            else:
                count += 1
                if count >= 5:
                    break
            if self.model.get_distance(x, self.params['final_goal'], self.params['weights'])< self.params['goal_radius']:
                best_x = x[0].copy()
                bestit = it +1
                min_loss = - np.inf
                break
            
            #self.cem.mu_u[:-self.params['control_dim']] = self.cem.mu_u[self.params['control_dim']:]
            #self.cem.sigma_u = self.cem.params['sigma_u']
        if self.verbose:
            print('min_loss:',min_loss)
        return best_x, min_loss, path[:best_it+1], collision, costs[:bestit+1]


