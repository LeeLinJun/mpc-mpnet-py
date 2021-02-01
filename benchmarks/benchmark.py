# from mpnet.sst_envs.utils import load_data, get_obs
import importlib
import click
import numpy as np
# import pickle
# import time
# from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py')


def full_benchmark(start_env,
                   num_env,
                   num_traj,
                   experiment_type,
                   save=True,
                   config='default',
                   report=True,
                   params=None,
                   system='acrobot_obs',
                   traj_id_offset=1800):

    sr = np.zeros((num_env, num_traj))
    planning_time = np.zeros((num_env, num_traj))
    costs = np.zeros((num_env, num_traj))

    for env_id in range(num_env):
        for traj_id in range(num_traj):
            experiment_func = importlib.import_module(".{}_exp".format(
                experiment_type), package="experiments").experiment
            result = experiment_func(
                env_id + start_env, traj_id + traj_id_offset, params=params, system=system)
            sr[env_id, traj_id] = result['successful']
            if result['successful']:
                planning_time[env_id, traj_id] = result['planning_time']
                costs[env_id, traj_id] = result['costs']
            if save:
                Path("results/{}/{}/{}/".format(
                    experiment_type, system, config)
                ).mkdir(parents=True, exist_ok=True)
                np.save('results/{}/{}/{}/sr_{}_{}.npy'.format(
                    experiment_type, system, config, num_env, num_traj), sr)
                np.save('results/{}/{}/{}/time_{}_{}.npy'.format(
                    experiment_type, system, config, num_env, num_traj),
                    planning_time)
                np.save('results/{}/{}/{}/costs_{}_{}.npy'.format(
                    experiment_type, system, config, num_env, num_traj),
                    costs)
            if report:
                sr_list = sr.reshape(-1)[:(num_traj*env_id+traj_id+1)]
                mask = sr_list > 0
                print("sr:{}\ttime:{}\tcosts:{}".format(sr_list.mean(),
                                                        planning_time.reshape(
                                                            -1)[:(num_traj*env_id+traj_id+1)][mask].mean(),
                                                        costs.reshape(-1)[:(num_traj*env_id+traj_id+1)][mask].mean()))


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
@click.option('--traj_id_offset', default=900)
@click.option('--experiment_type', default="shm")
def main(full, env_id, traj_id, num_env, num_traj, save, config, report, system, traj_id_offset, experiment_type):
    if full is not True:
        experiment_func = importlib.import_module(".{}_exp".format(
            experiment_type), package="experiments").experiment
        result = experiment_func(env_id, traj_id,
                                 params=importlib.import_module('.{}_{}'.format(
                                     experiment_type, config), package=".params.{}".format(system)).get_params(),
                                 system=system)
        # print(result)
        Path('results/traj/{system}/{experiment_type}/{env_id}/'.format(
            system=system, experiment_type=experiment_type, env_id=env_id, traj_id=traj_id)
        ).mkdir(parents=True, exist_ok=True)
        np.save('results/traj/{system}/{experiment_type}/{env_id}/env_{env_id}_id_{traj_id}.npy'.format(
            system=system, experiment_type=experiment_type, env_id=env_id, traj_id=traj_id), result['traj'])

    else:
        result = full_benchmark(env_id,
                                num_env,
                                num_traj,
                                experiment_type,
                                save=save,
                                config=config,
                                report=report,
                                params=importlib.import_module('.{}_{}'.format(
                                    experiment_type, config), package=".params.{}".format(system)).get_params(),
                                system=system,
                                traj_id_offset=traj_id_offset)


if __name__ == '__main__':
    main()
