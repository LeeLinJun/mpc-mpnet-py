import numpy as np
import pickle
import time
import click
from tqdm import tqdm
from pathlib import Path
import importlib
from mpnet.sst_envs.utils import load_data, get_obs

def experiment(env_id, traj_id, verbose=False, system='acrobot_obs', params=None):
    print("env {}, traj {}".format(env_id, traj_id))
    data = load_data(system, env_id, traj_id)
    # ref_path = data['path']
    # start_goal = data['start_goal']
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
