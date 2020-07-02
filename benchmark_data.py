import numpy as np
import pickle
import time
import click
from tqdm import tqdm
from pathlib import Path
import importlib
from mpnet.sst_envs.utils import load_data, get_obs

def experiment(env_id, traj_id, verbose=False, system='acrobot_obs', params_module=None):
    print("env {}, traj {}".format(env_id, traj_id))
    data = load_data(system, env_id, traj_id)
    ref_path = data['path']
    start_goal = data['start_goal']
    costs = data['cost'].sum()
#     print(mpc_mpnet.costs)
    result = {
        'env_id': env_id,
        'traj_id': traj_id,
        'planning_time': 0,
        'successful': 1,
        'costs': costs
    }
    
    print("\t{}, time: {} seconds, {}(ref:{}) costs".format(
            result['successful'],
            result['planning_time'],
            result['costs'],
            data['cost'].sum()))
    return result

def full_benchmark(num_env, num_traj, save=True, config='default', report=True, params_module=None, system='acrobot_obs'):
    sr = np.zeros((num_env, num_traj))
    time = np.zeros((num_env, num_traj))
    costs = np.zeros((num_env, num_traj))

    for env_id in range(num_env):
        for traj_id in range(num_traj):
            result = experiment(env_id, traj_id, params_module=params_module, system=system)
            sr[env_id, traj_id] = result['successful']
            if result['successful']:
                time[env_id, traj_id] = result['planning_time']
                costs[env_id, traj_id] = result['costs']
            if save:
                Path("results/cpp_full/{}/{}/".format(config, system)).mkdir(parents=True, exist_ok=True)
                np.save('results/cpp_full/{}/{}/sr_{}_{}.npy'.format(config, system, num_env, num_traj), sr)
                np.save('results/cpp_full/{}/{}/time_{}_{}.npy'.format(config, system, num_env, num_traj), time)
                np.save('results/cpp_full/{}/{}/costs_{}_{}.npy'.format(config, system,  num_env, num_traj), costs)
            if report:
                print("sr:{}\ttime:{}\tcosts:{}".format(
                    sr.reshape(-1)[:(num_traj*env_id+traj_id+1)].mean(),
                    time.reshape(-1)[:(num_traj*env_id+traj_id+1)].mean(),
                    costs.reshape(-1)[:(num_traj*env_id+traj_id+1)].mean(),
                    ))

@click.command()
@click.option('--full', default=True)
@click.option('--env_id', default=0)
@click.option('--traj_id', default=0)
@click.option('--num_env', default=10)
@click.option('--num_traj', default=100)
@click.option('--save', default=True)
@click.option('--config', default='default')
@click.option('--report', default=True)
@click.option('--system', default="cartpole_obs")
def main(full, env_id, traj_id, num_env, num_traj, save, config, report, system):
    p = importlib.import_module('.cpp_dst_{}'.format(config), package=".params.{}".format(system))
    if not full:
        result = experiment(env_id, traj_id, system=system)
    else:
        result = full_benchmark(num_env, num_traj, save, config, report, p, system=system)
   
    
    
if __name__ == '__main__':
    main()
