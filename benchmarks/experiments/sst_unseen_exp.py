from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
from mpnet.sst_envs.utils import load_data, get_obs_unseen, get_obs_3d_unseen
import time
from tqdm import tqdm

import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')

def experiment(env_id, traj_id, verbose=False, system='cartpole_obs', params=None):
    # for unseen obs
    print("env {}, traj {}".format(env_id, traj_id))
    if system in ['cartpole_obs', 'acrobot_obs', 'car_obs']:
        obs_list = get_obs_unseen(system)[env_id]
    elif system in ['quadrotor_obs']:
        obs_list = get_obs_3d_unseen(system)[env_id].reshape(-1, 3)
    data = load_data(system, env_id, traj_id)
    # ref_path = data['path']
    ref_cost = data['cost'].sum()
    start_goal = data['start_goal']
    min_time_steps = params['min_time_steps']
    max_time_steps = params['max_time_steps']
    integration_step = params['integration_step']

    if system == 'quadrotor_obs':
        env_constr = standard_cpp_systems.RectangleObs3D
        env = env_constr(obs_list, params['width'], 'quadrotor')
    else:
        print("unkown system")
        exit(-1)
    planner = _sst_module.SSTWrapper(
        state_bounds=env.get_state_bounds(),
        control_bounds=env.get_control_bounds(),
        distance=env.distance_computer(),
        start_state=start_goal[0],
        goal_state=start_goal[-1],
        goal_radius=params['goal_radius'],
        random_seed=params['random_seed'],
        sst_delta_near=params['sst_delta_near'],
        sst_delta_drain=params['sst_delta_drain']
    )
    solution = planner.get_solution()

    # data_cost = np.sum(data['cost'])
    # th = 1.2 * data_cost
    # # start experiment
    tic = time.perf_counter()
    for iteration in tqdm(range(params['number_of_iterations'])):
        planner.step(env,
                     min_time_steps,
                     max_time_steps,
                     integration_step)
        if iteration % 100 == 0:
            solution = planner.get_solution()
            if (solution is not None and solution[2].sum() <= ref_cost*1.4) or time.perf_counter()-tic > params['max_planning_time']:
                # and np.sum(solution[2]) < th:
                break
    toc = time.perf_counter()

    costs = solution[2].sum() if solution is not None else np.inf
    result = {
        'env_id': env_id,
        'traj_id': traj_id,
        'planning_time': toc-tic,
        'successful': True,
        'costs': costs
    }

    print("\t{}, time: {} seconds, {}(ref:{}) costs".format(
        result['successful'],
        result['planning_time'],
        result['costs'],
        np.sum(data['cost'])))
    return result
