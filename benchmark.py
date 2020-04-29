import numpy as np
import torch
from mpc_mpnet_planner import MPCMPNetPlanner
# from matplotlib import pyplot as plt
from mpnet.sst_envs.utils import load_data, visualize_point, get_obs
import pickle

import time
import click


def get_params(final_goal):
    n_t = 5
    state_dim = 4
    control_dim = 1
    weights = np.ones(state_dim)*1
    weights[2:] = 0.25
    n_sample = 1024
    n_elite = 32
    t = 1e-1
    dt = 2e-2

    mu_t, sigma_t = 1e-1, 4e-1
    t_min, t_max = 0, 5e-1

    mu_u = np.zeros((n_t*control_dim))
    sigma_u_diag = np.ones(n_t*control_dim)
    sigma_u_diag[:] = 4
    sigma_u = np.diag(sigma_u_diag)
    params = {
        'n_sample': n_sample,
        'n_elite': n_elite,
        'n_t': n_t,
        'weights': weights,
        'mu_u': mu_u,
        'sigma_u': sigma_u,
        't': t,
        'dt': dt,

        'mu_t': np.ones(n_t) * mu_t,
        'sigma_t': np.identity(n_t)*sigma_t,
        't_min': t_min,
        't_max': t_max,

        'state_dim': state_dim,
        'control_dim': control_dim,
        'converge_radius': 1e-2,
        'drop_radius': 1,
        'goal_radius': 10, #np.sqrt(2),
        'max_it': 20,
        'rolling_count': n_t,
        'bk_it': 2,
        'final_goal': final_goal, #ref_path[-1], #np.array([np.inf, np.inf, np.inf, np.inf]), #
        'mpc_mode': 'solve',
    #     'mpc_mode': 'rolling',
    }
    return params
    
def experiment(env_id, traj_id, verbose=False, max_it=300):
    obs_list = get_obs('acrobot_obs', env_id)
    data = load_data('acrobot_obs', env_id, traj_id)
    ref_path = data['path']
    env_vox = torch.from_numpy(np.load('mpnet/sst_envs/acrobot_obs_env_vox.npy')[env_id]).unsqueeze(0).float()
    
    tic = time.perf_counter()

    ## initiate planner
    mpc_mpnet = MPCMPNetPlanner(
        get_params(ref_path[-1]),
        ref_path[0], 
        ref_path[-1],
        env_vox,
        system="acrobot_obs",
        #setup="default_norm",
        #setup="default_norm_aug",
        setup="norm_nodiff_noaug_20step2e-2",
        ep=5000,
        obs_list=obs_list[env_id], 
        verbose=verbose)
    mpc_mpnet.mpnet.train()
    
    ## start experiment
    it = 0
    while it < max_it and not mpc_mpnet.reached:
        mpc_mpnet.reset()
        for i in range(50):
            if verbose:
                print('iteration:', i)
            mpc_mpnet.plan()
            if mpc_mpnet.reached:
                break
            it += 1
    
    toc = time.perf_counter()
#     print(mpc_mpnet.costs)
    
    result = {
        'env_id': env_id,
        'traj_id': traj_id,
        'planning_time': toc-tic,
        'successful': mpc_mpnet.reached,
        'costs': np.sum(np.reshape(mpc_mpnet.costs, -1)[0]) if len(mpc_mpnet.costs) > 0 else np.inf
             }
    
    print("env {}, traj {}, {}, time: {} seconds".format(
        env_id, 
        traj_id,
        result['successful'],
        result['planning_time'],
        ))
    return result
    

def full_benchmark(num_env, num_traj):
    sr = np.zeros((num_env, num_traj))
    time = np.zeros((num_env, num_traj))
    costs = np.zeros((num_env, num_traj))

    for env_id in range(num_env):
        for traj_id in range(num_traj):
            result = experiment(env_id, traj_id)
            sr[env_id, traj_id] = result['successful']
            if result['successful']:
                time[env_id, traj_id] = result['planning_time']
                costs[env_id, traj_id] = result['costs']
            np.save('results/full/sr_{}_{}.npy'.format(num_env, num_traj), sr)
            np.save('results/full/time_{}_{}.npy'.format(num_env, num_traj), time)
            np.save('results/full/costs_{}_{}.npy'.format(num_env, num_traj), costs)


@click.command()
@click.option('--full', default=True)
@click.option('--env_id', default=0)
@click.option('--traj_id', default=0)
@click.option('--num_env', default=10)
@click.option('--num_traj', default=1000)
def main(full, env_id, traj_id, num_env, num_traj):
    if not full:
        result = experiment(env_id, traj_id)
    else:
        result = full_benchmark(num_env, num_traj)
   
    
    
if __name__ == '__main__':
    main()