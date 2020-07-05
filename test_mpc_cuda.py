import numpy as np
from mpnet.sst_envs.utils import load_data, get_obs
import pickle
import time
import click
from tqdm.auto import tqdm
from pathlib import Path
import importlib
from matplotlib import pyplot as plt

import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')

from sparse_rrt import _deep_smp_module



def experiment_vis(env_id, traj_id, verbose=False, system='cartpole_obs', params=None, ax=None, bx=None, number_of_iterations=1):
    print("env {}, traj {}".format(env_id, traj_id))
    obs_list = get_obs(system, env_id)[env_id].reshape(-1, 2)
    data = load_data(system, env_id, traj_id)
    ref_path = data['path']
    start_goal = data['start_goal']
    # print(start_goal)
    env_vox = np.load('mpnet/sst_envs/{}_env_vox.npy'.format(system))
    obc = env_vox[env_id, 0]
    # print(obs_list)
    params = params
    #number_of_iterations = params['number_of_iterations'] #3000000# 
    min_time_steps = params['min_time_steps'] if 'min_time_steps' in params else 80
    max_time_steps = params['max_time_steps'] if 'min_time_steps' in params else 400
    integration_step = params['dt']
    
    planner = _deep_smp_module.DSSTMPCWrapper(
        system_type=system,
        solver_type="cem_cuda",
        start_state=np.array(ref_path[0]),
    #             goal_state=np.array(ref_path[-1]),
        goal_state=np.array(data['start_goal'][-1]),
        goal_radius=params['goal_radius'],
        random_seed=0,
        sst_delta_near=params['sst_delta_near'],
        sst_delta_drain=params['sst_delta_drain'],
        obs_list=obs_list,
        width=params['width'],
        verbose=params['verbose'],
        mpnet_weight_path=params['mpnet_weight_path'], 
        cost_predictor_weight_path=params['cost_predictor_weight_path'],
        cost_to_go_predictor_weight_path=params['cost_to_go_predictor_weight_path'],
        num_sample=params['cost_samples'],
        np=2, ns=params['n_sample'], nt=params['n_t'], ne=params['n_elite'], max_it=params['max_it'],
        converge_r=params['converge_r'], mu_u=params['mu_u'], std_u=params['sigma_u'], mu_t=params['mu_t'], 
        std_t=params['sigma_t'], t_max=params['t_max'], step_size=params['step_size'], integration_step=params['dt'], 
        device_id=params['device_id'], refine_lr=params['refine_lr'],
        weights_array=params['weights_array'],
        obs_voxel_array=obc.reshape(-1)
    )
    return data, planner
    


env_id = 0
traj_id = 1801
system = 'cartpole_obs'
config = 'default'

params = {
        'n_sample': 256,
        'n_elite': 4,
        'n_t': 4,
        'max_it': 5,
        'converge_r': 1e-3,
        
        'dt': 2e-3,

        'mu_u': 0,
        'sigma_u': 300,

        'mu_t': 0.025,
        'sigma_t': 0.05,
        't_max': 0.1,

        'verbose': True, #False,#
        'step_size': 0.75,

        "goal_radius": 1.5,

        "sst_delta_near": 2,
        "sst_delta_drain": 0.1,
        "goal_bias": 0.1,

        "width": 4,
        "hybrid": False,#True,#
        "hybrid_p": 0,#1,

        "cost_samples": 20,
        "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_pos_vel_external.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v3_multigoal.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v2_deep.pt",
#         "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",
#         "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_branch.pt",


        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

        "cost_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_10k.pt",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 500,
        "weights_array": [1, 1, 1, 1],

    }

data, planner = experiment_vis(env_id, 
                    traj_id, 
                    verbose=False, 
                    system='cartpole_obs', 
                    params=params, 
                    ax=None, 
                    bx=None,
                    number_of_iterations=params['number_of_iterations'])


ref_path = data['path']
import time

state = ref_path[0]
full_time = 0
#for i in [0, 1, 2]:
for i in range(len(ref_path)-1):
    tic = time.perf_counter()
#     state, sample, actual_state = ref_path[i], ref_path[i+1], planner.steer(ref_path[i], ref_path[i+1])
    sample = ref_path[i+1].copy()
    actual_state = planner.steer_batch(np.array([state.copy(), state.copy()]), np.array([sample.copy(), sample.copy()]), 2)
    print(actual_state)
#     print(state, sample, actual_state)
#     toc = time.perf_counter()
#     time_cost = toc - tic
#     full_time += time_cost
#     print(time_cost)
#     visualize_pair(state, sample, actual_state, ax, bx)
#     state = actual_state.copy()
print("total: {}".format(full_time))
