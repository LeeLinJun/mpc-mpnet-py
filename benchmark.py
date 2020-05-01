import numpy as np
import torch
from mpc_mpnet_planner import MPCMPNetPlanner
# from matplotlib import pyplot as plt
from mpnet.sst_envs.utils import load_data, visualize_point, get_obs
import pickle

import time
import click

from params.sst_step5_s1024_e32 import get_params

def experiment(env_id, traj_id, verbose=False):
    obs_list = get_obs('acrobot_obs', env_id)
    data = load_data('acrobot_obs', env_id, traj_id)
    ref_path = data['path']
    env_vox = torch.from_numpy(np.load('mpnet/sst_envs/acrobot_obs_env_vox.npy')[env_id]).unsqueeze(0).float()
    
    tic = time.perf_counter()
    
    ## initiate planner
    params = get_params(ref_path[-1])
    mpc_mpnet = MPCMPNetPlanner(
        params,
        ref_path[0], 
        ref_path[-1],
        env_vox,
        system="acrobot_obs",
        setup="default_norm",
        #setup="default_norm_aug",
#         setup="norm_nodiff_noaug_20step2e-2",
        ep=5000,
        obs_list=obs_list[env_id], 
        verbose=verbose)
    mpc_mpnet.mpnet.train()
    
    ## start experiment
    it = 0
    while it < params['max_plan_it'] and not mpc_mpnet.reached:
        mpc_mpnet.reset()
        for i in range(20):
            if verbose:
                print('iteration:', i)
            mpc_mpnet.step()
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
    

def full_benchmark(num_env, num_traj, save=True, config='default'):
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
            if save:
                np.save('results/full/sr_{}_{}_{}.npy'.format(config, num_env, num_traj), sr)
                np.save('results/full/time_{}_{}_{}.npy'.format(config, num_env, num_traj), time)
                np.save('results/full/costs_{}_{}_{}.npy'.format(config, num_env, num_traj), costs)


@click.command()
@click.option('--full', default=True)
@click.option('--env_id', default=0)
@click.option('--traj_id', default=0)
@click.option('--num_env', default=10)
@click.option('--num_traj', default=1000)
@click.option('--save', default=True)
@click.option('--config', default='ls')
def main(full, env_id, traj_id, num_env, num_traj, save, config):
    if not full:
        result = experiment(env_id, traj_id)
    else:
        result = full_benchmark(num_env, num_traj, save, config)
   
    
    
if __name__ == '__main__':
    main()