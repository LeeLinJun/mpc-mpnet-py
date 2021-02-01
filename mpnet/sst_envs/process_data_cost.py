import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')

from utils import load_data, get_obs
import numpy as np
import click

# from systems.acrobot import Acrobot
from systems.quadrotor import Quadrotor
from sparse_rrt.systems.car import Car
from sparse_rrt.systems.cartpole import Cartpole
from sparse_rrt.systems.acrobot import Acrobot


from pathlib import Path
from tqdm import tqdm


def get_dynamics(dynamics):
    sys_dict = {
        "acrobot_obs": "Acrobot",
        "cartpole_obs": "Cartpole",
        "quadrotor_obs": "Quadrotor",
        "car_obs": "Car"
    }
    return eval("{}()".format(sys_dict[dynamics]))

def post_propagate(path_dict, system, dt=2e-3, interp_freq=1):
    dynamics = get_dynamics(system)
    ref_path = path_dict['path']
    ref_control = path_dict['control']
    ref_time = path_dict['cost']

    post_state = ref_path[0].copy().astype(np.float)
    post_propagate_path = [post_state.copy()]
    counter = 0
    post_propagate_cost = []
    cost = 0

    for i in range(ref_control.shape[0]):
        num_steps = int(ref_time[i] / dt)
        for step in range(num_steps):
            post_state = dynamics.propagate(post_state,
                                            ref_control[i],
                                            1,
                                            dt)
            counter += 1
            cost += dt
            if counter % interp_freq == 0:
                post_propagate_path.append(post_state.copy())
                post_propagate_cost.append(cost)
                cost = 0
                counter = 0
    post_propagate_path.append(post_state.copy())
    post_propagate_cost.append(cost)
    return np.array(post_propagate_path), np.array(post_propagate_cost)


def path_to_tensor_forward(env_id,
                           path_dict,
                           normalize,
                           interpolate=False,
                           interpolate_steps=20,
                           system="acrobot_obs",
                           step_size=2e-3,
                           goal_aug=False):
    """
    [env_id, state, goal]
    """
    # print(goal_aug)
    if interpolate:
        path, costs = post_propagate(path_dict, system, step_size, interp_freq=interpolate_steps)
    # print(np.sum(path_dict['cost']), costs, np.sum(costs))

    # start_goal = path_dict['start_goal']
    n_nodes = path.shape[0]
    # state_size = path.shape[1]
    data = []
    gt = []
    c2g = []

    if system == 'quadrotor_obs':
        for i in range(len(path)):
            if path[i, 6] < 0:
                path[i, 3:7] *= -1

    # start to first path node
    for i_start in range(n_nodes-1):
        goal_ind_list = range(i_start+1, n_nodes) if goal_aug else [n_nodes - 1]
        for i_goal in goal_ind_list:
            data.append(np.concatenate(([env_id],
                                        path[i_start, :],
                                        path[i_goal, :])))
            gt.append(path[i_start+1, :])
            c2g.append(np.sum(costs[i_start:i_goal+1]))
    # print(np.max(c2g), np.sum(path_dict['cost']))

    data = np.array(data)
    gt = np.array(gt)
    c2g = np.array(c2g)
    if normalize:
        if system == "acrobot_obs":
            data[:, [1, 2, 5, 6]] /= np.pi
            data[:, [3, 4, 7, 8]] /= 6
            gt[:, [0, 1]] /= np.pi
            gt[:, [2, 3]] /= 6
        elif system == "cartpole_obs":
            data[:, 1:] /= np.array([30, 40, np.pi, 2, 30, 40, np.pi, 2])
            gt /= np.array([30, 40, np.pi, 2])
        elif system == 'quadrotor_obs':
            data[:, 1:] /= np.array([  # start
                5, 5, 5,
                1, 1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                # goal
                5, 5, 5,
                1, 1, 1, 1,
                1, 1, 1,
                1, 1, 1])
            gt /= np.array([5, 5, 5,
                            1, 1, 1, 1,
                            1, 1, 1,
                            1, 1, 1])
        elif system == 'car_obs':
            data[:, 1:] /= np.array([25, 25, np.pi, 25, 25, np.pi])
            gt /= np.array([25, 25, np.pi])
        else:
            raise NotImplementedError("unkown dynamics")
    return data, gt, c2g


@click.command()
@click.option('--num', default=10)
@click.option('--system', default='quadrotor_obs')
@click.option('--traj_num', default=900)
@click.option('--setup', default='default_norm')
@click.option('--normalize', default=True)
@click.option('--interpolate', default=True)
@click.option('--interpolate_steps', default=20)
@click.option('--step_size', default=2e-3)
@click.option('--goal_aug', default=False)
def main(num, system, traj_num, setup, normalize, interpolate,
         interpolate_steps, step_size, goal_aug):
    print("goal_aug is {}".format(goal_aug))
    print("Interpolating is set to {}, step_size is {} and interpolate to {} steps".format(interpolate, step_size, interpolate_steps))
    data, gt, cost_to_go = [], [], []
    for env_id in range(num):
        # infeasible_points = get_obs_states(env_id = env_id, system=system)
        for traj_id in tqdm(range(traj_num)):
            # try:
            path_dict = load_data(system, env_id, traj_id)
            data_lists = path_to_tensor_forward(env_id,
                                                path_dict,
                                                normalize,
                                                interpolate=interpolate,
                                                interpolate_steps=interpolate_steps,
                                                system=system,
                                                step_size=step_size,
                                                goal_aug=goal_aug)
            d, g, c2g = data_lists
            # print(c2g, np.sum(path_dict['cost']))
            data.append(d)
            gt.append(g)
            cost_to_go.append(c2g) 
    data = np.concatenate(data, axis=0)
    gt = np.concatenate(gt, axis=0)
    cost_to_go = np.concatenate(cost_to_go, axis=0)
    print([d.shape for d in [data, gt, cost_to_go]])

    # print(cost_to_go[:100])

    Path("data/{}".format(setup)).mkdir(parents=True, exist_ok=True)
    np.save('data/{}/{}_path_data.npy'.format(setup, system), data)
    np.save('data/{}/{}_gt.npy'.format(setup, system), gt)
    np.save('data/{}/{}_cost_to_go.npy'.format(setup, system), cost_to_go)


if __name__ == '__main__':
    main()
