import numpy as np
from mpnet.sst_envs.utils import load_data, get_obs, get_obs_3d
import time
from tqdm import tqdm

import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')
from sparse_rrt import _mpc_mpnet_module

def experiment(env_id, traj_id, verbose=False, system='cartpole_obs', params=None):
    print("env {}, traj {}".format(env_id, traj_id))
    # print(params)
    if system in ['cartpole_obs', 'acrobot_obs', 'car_obs']:
        obs_list = get_obs(system)[env_id].reshape(-1, 2)
    elif system in ['quadrotor_obs']:
        obs_list = get_obs_3d(system)[env_id].reshape(-1, 3)
    data = load_data(system, env_id, traj_id)
    # ref_path = data['path']
    # start_goal = data['start_goal']
    # print(start_goal)
    env_vox = np.load('mpnet/sst_envs/data/{}_env_vox.npy'.format(system))
    obc = env_vox[env_id]
    # print(obc.reshape(-1), obc.reshape(-1).shape)
    # print(obs_list)
    number_of_iterations = params['number_of_iterations']  # 3000000#
    # min_time_steps = params['min_time_steps'] if 'min_time_steps' in params else 0
    # max_time_steps = params['max_time_steps'] if 'min_time_steps' in params else 800
    # integration_step = params['dt']
    planner = _mpc_mpnet_module.MPCMPNetWrapper(system_type=system,
                                                solver_type=params['solver_type'],
                                                start_state=np.array(
                                                    data['start_goal'][0]),
                                                goal_state=np.array(
                                                    data['start_goal'][1]),
                                                goal_radius=params['goal_radius'],
                                                random_seed=0,
                                                sst_delta_near=params['sst_delta_near'],
                                                sst_delta_drain=params['sst_delta_drain'],
                                                obs_list=obs_list,
                                                width=params['width'],
                                                verbose=params['verbose'],
                                                mpnet_weight_path=params['mpnet_weight_path'],
                                                cost_predictor_weight_path=params['cost_predictor_weight_path'],
                                                cost_to_go_predictor_weight_path=params[
                                                    'cost_to_go_predictor_weight_path'],
                                                num_sample=params['cost_samples'],
                                                # shm_max_step=params['shm_max_steps'],
                                                np=params['n_problem'], ns=params['n_sample'], nt=params[
                                                    'n_t'], ne=params['n_elite'], max_it=params['max_it'],
                                                converge_r=params['converge_r'], mu_u=params[
                                                    'mu_u'], std_u=params['sigma_u'], mu_t=params['mu_t'],
                                                std_t=params['sigma_t'], t_max=params['t_max'], step_size=params[
                                                    'step_size'], integration_step=params['dt'],
                                                device_id=params['device_id'], refine_lr=params['refine_lr'],
                                                weights_array=params['weights_array'],
                                                obs_voxel_array=obc.reshape(-1)
                                                )
    solution = planner.get_solution()

    # data_cost = np.sum(data['cost'])
    # th = 1.2 * data_cost
    # start experiment
    tic = time.perf_counter()
    for iteration in tqdm(range(number_of_iterations)):
        planner.mp_tree_step(params['refine'],
                             refine_threshold=params['refine_threshold'],
                             using_one_step_cost=params['using_one_step_cost'],
                             cost_reselection=params['cost_reselection'],
                             goal_bias=params['goal_bias'],
                             num_of_problem=params['n_problem'])
        solution = planner.get_solution()
        # and np.sum(solution[2]) < th:
        if solution is not None or time.perf_counter()-tic > params['max_planning_time']:
            break
    toc = time.perf_counter()
    # if solution is not None:
    #     print(solution[0], solution[2])
#     print(mpc_mpnet.costs)
    costs = solution[2].sum() if solution is not None else np.inf
    result = {
        'env_id': env_id,
        'traj_id': traj_id,
        'planning_time': toc-tic,
        'successful': solution is not None,
        'costs': costs,
        'traj': solution[0]
    }

    print("\t{}, time: {} seconds, {}(ref:{}) costs".format(
        result['successful'],
        result['planning_time'],
        result['costs'],
        np.sum(data['cost'])))
    return result
