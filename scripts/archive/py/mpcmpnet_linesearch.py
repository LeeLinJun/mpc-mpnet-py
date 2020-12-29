import torch
import numpy as np

from mpnet.networks.mpnet import MPNet
from mpnet.normalizer import Normalizer
from mpc.cem_mpc_teb import MPC
from mpc.systems.acrobot_vec_tv import Acrobot

class MPCMPNetPlanner:
    def __init__(self,
                 params,
                 start, # only one start in [state_dim]
                 goal, # only one goal in [state_dim]
                 env_vox,
                 system="acrobot_obs",
                 setup="norm_nodiff_noaug_20step2e-2",
                 ep=5,
                 obs_list=None,
                 verbose=False
                ):
        self.verbose = verbose
        self.system = system
        
        self.mpnet = MPNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=params['state_dim'])
        self.mpnet.load_state_dict(torch.load('mpnet/output/{}/{}/ep{}'.format(system, setup, ep)))
        # enable dropout randomness
        self.mpnet.train()
        
        self.dynamics = Acrobot()
        self.params = params
        self.mpc = MPC(self.params, self.dynamics, verbose=self.verbose)
        self.normalizer = Normalizer(self.system)
        
        self.obs_list = obs_list
        self.start = start
        self.goal = np.expand_dims(goal, axis=0)
        self.env_vox = env_vox     
        
        self.reset()
        
        self.trials = []
        
    def reset(self):
        self.state = np.expand_dims(self.start, axis=0)
        self.bk_state = [self.state.copy()]
        self.bk_count = 0
        self.path = []
        self.samples = []
        self.reached = False
        self.costs = []
        
    def sample(self, start, goal):
        """sample a way point
        
        Arguments:
            start {[float]} -- np.array in [:, state_dim]
            goal {[float]} -- np.array in [:, state_dim]
        
        Returns:
            sample -- np.array in [:, state_dim]
        """

        start_th = torch.from_numpy(self.normalizer.normalize(start))
        goal_th = torch.from_numpy(self.normalizer.normalize(goal))
        
        start_goal = torch.cat((start_th.float(), goal_th.float()), dim=1)
        with torch.no_grad():
            sample = self.mpnet(start_goal, self.env_vox) # in normalized space
        
        sample = self.normalizer.denormalize(sample.numpy())
        return sample # in configration space
    
    def steer(self, sample):
        """steer nodes with mpc
        
            Arguments:
                sample {[float]} -- numpy.array in [state_dim] as goal point
                
            Returns:
                if not collision and converge:
                    [best_x:np.array, min_loss:float, path:list, collision:bool]
                    where:
                        best_x in [state_dim]
                else None
        """
        if self.params['mpc_mode'] == 'rolling':
            best_x, min_loss, path, collision = self.mpc.rolling(self.state[0], sample, obs_list=self.obs_list)
        elif self.params['mpc_mode'] == 'solve':
            best_x, min_loss, path, collision, cost, _ = self.mpc.solve(self.state[0], sample, obs_list=self.obs_list)
#         print(self.state[0])    
#         print(path)
     
        if collision or min_loss > self.params['drop_radius']:
            if self.verbose:
                print('\t drop', min_loss)
            self.bk_count += 1
            if self.bk_count > self.params['bk_it']:
                self.bk_count = 0
#                 self.reset()
#                 self.trials.append(self.path.copy())
                self.state = self.bk_state[-1]
                self.bk_state.pop()
            return [], [], False
        else:
            self.bk_count = 0
            self.state = np.expand_dims(best_x, axis=0).copy()
            self.bk_state += [self.state.copy()]
            return path, np.sum(np.reshape(cost, -1)), True # new state and path
        
    def plan(self):
        """plan if not converge, firstly sample a way point, then try to steer with mpc
           
           Arguments:
                None
                
           Returns:
                None
        """
        if not self.reached:
            goal_distance = self.dynamics.get_distance(self.state, self.goal[0], self.params['weights'])
#             goal_distance = self.dynamics.get_loss(self.state, self.goal[0], self.params['weights'])

            if  goal_distance < self.params['goal_radius']:
                print('####reached####\ngoal distance:', goal_distance)
                self.reached = True
                return
            else:
                if self.verbose:
                    print(goal_distance)
            
            #if goal_distance > 10:
            if True:
                sample = self.sample(self.state, self.goal)
                self.samples.append(sample[0])
            else:
                sample = self.goal
            ## pass sample in 1-dim [state_dim]
            path_i, cost_i, success = self.steer(sample[0])
            if success:
                self.path += path_i
                self.costs += [cost_i]
            
#     def plan_waypoints(self):
#         """plan if not converge, sample waypoints to form a path
           
#            Arguments:
#                 None
                
#            Returns:
#                 None
#         """
#         if not self.reached:
#             goal_distance = self.dynamics.get_distance(self.state, self.goal[0], self.params['weights'])

#             if  goal_distance < self.params['goal_radius']:
#                 if self.verbose:
#                     print('reached, goal distance:', goal_distance)
#                 self.reached = True
#                 return
#             else:
#                 if self.verbose:
#                     print(goal_distance)
            
#             if goal_distance > 3:
#                 sample = self.sample(self.state, self.goal)
#                 self.samples.append(sample[0])
#             else:
#                 sample = self.goal
           
#             ## pass sample in 1-dim [state_dim]
# #             path_i = self.steer(sample[0])
#             self.state = sample.copy()
#             self.path.append(sample[0].copy())
        
        
        
        