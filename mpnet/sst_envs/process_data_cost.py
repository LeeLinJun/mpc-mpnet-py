from utils import load_data, get_obs
import numpy as np
import pickle
import re
import click
from systems.acrobot import Acrobot
from systems.quadrotor import QuadrotorVec

from pathlib import Path
from tqdm import tqdm

def get_dynamics(dynamics):
    sys_dict = {
        "acrobot_obs": "Acrobot",
        "cartpole_obs": "CartPole",
        "quadrotor_obs": "QuadrotorVec"
    }
    return eval("sys_dict[dynamics]()")



def get_obs_states(env_id, system='cartpole_obs',):
    if system == 'cartpole_obs':
        def line_line_cc(x1, y1, x2, y2, x3, y3, x4, y4):
            uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            if uA >= 0. and uA <= 1. and uB >= 0. and uB <= 1.:
                # intersection
                return True
            # collision free
            return False
        def IsInCollision(x, obc, obc_width=4.):
            I = 10
            L = 2.5
            M = 10
            m = 5
            g = 9.8
            H = 0.5

            STATE_X = 0
            STATE_V = 1
            STATE_THETA = 2
            STATE_W = 3
            CONTROL_A = 0

            MIN_X = -30
            MAX_X = 30
            MIN_V = -40
            MAX_V = 40
            MIN_W = -2
            MAX_W = 2


            if x[0] < MIN_X or x[0] > MAX_X:
                return True
            H = 0.5
            pole_x1 = x[0]
            pole_y1 = H
            pole_x2 = x[0] + L * np.sin(x[2])
            pole_y2 = H + L * np.cos(x[2])

            width = 4
            for i in range(len(obc)):
                for j in range(0, 8, 2):
                    x1 = obc[i][j]
                    y1 = obc[i][j+1]
                    x2 = obc[i][(j+2) % 8]
                    y2 = obc[i][(j+3) % 8]
                    if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                        return True
            return False
        _obs_list = get_obs(system, env_id)[env_id]#.reshape(-1, 2)
        obs_list = []
        width = 4
        for i in range(len(_obs_list)):
            x = _obs_list[i][0];
            y = _obs_list[i][1];
            obs = np.zeros(8)
            obs[0] = x - width / 2;  obs[1] = y + width / 2;
            obs[2] = x + width / 2;  obs[3] = y + width / 2;
            obs[4] = x + width / 2;  obs[5] = y - width / 2;
            obs[6] = x - width / 2;  obs[7] = y - width / 2;
            obs_list.append(obs)
        obs_i = np.array(obs_list)
        dx = 5
        dtheta = 0.5
        # feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*30./dx)
        jmin = 0
        jmax = int(2*np.pi/dtheta)

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                x = np.array([dx*i-30, 0., dtheta*j-np.pi, 0.])
                if IsInCollision(x, obs_i):
                    infeasible_points.append(x)
                    # pass
                # else:
                    # feasible_points.append(x)
        # feasible_points = np.array(feasible_points)
        infeasible_points = np.array(infeasible_points)
        # print('feasible points')
        # print(feasible_points)
        # print('infeasible points')
        # print(infeasible_points)

    elif system == 'acrobot_obs':
        return None
    else:
        raise NotImplementedError("unkown dynamics")        

    return infeasible_points

def interpolate_path(path_dict, system="acrobot_obs", interval_steps=20, step_size=2e-2):
    dynamics = get_dynamics(system)
    ref_path = path_dict['path']
    ref_control = path_dict['control']
    ref_time = path_dict['cost']
    state = ref_path[0].copy()
    waypoints = [state].copy()
    cost = 0
    costs = [cost]
    for k in range(len(ref_control)):
        max_steps = int(np.round(ref_time[k]/step_size))
        #print(state, ref_path[k])
        state = ref_path[k].copy()
        waypoints[-1] = ref_path[k]
        for step in range(1, max_steps+1):
            state = dynamics.propagate(state.copy(), [ref_control[k]], 1, step_size).copy()
            cost += step_size
            if (step % interval_steps == 0) or (step == max_steps):
                waypoints.append(state.copy())
                costs.append(cost)
                #print(step, state)
    waypoints[-1] = ref_path[-1].copy()
    #print(waypoints[-1], ref_path[-1])
    waypoints = np.array(waypoints)
    costs = np.array(costs)
    costs_sofar = costs.copy()
    costs2go = cost - costs
#     print(costs2go)
#     print(waypoints.shape, costs2go.shape)
    #print(waypoints) 
    return waypoints, costs_sofar, costs2go

def post_propagate(start_state, solution, dt=2e-2, interp_freq=1):
    assert solution is not None
    system = QuadrotorVec()
    post_state = start_state.copy().astype(np.float)
    post_propagate_path = [post_state.copy()]
    counter = 0
    for i in range(solution[1].shape[0]):
        num_steps = int(solution[2][i] / 2e-2)
        for step in range(num_steps):
            post_state = system.propagate(post_state,
                                          solution[1][i],
                                          1,
                                          dt)
            counter += 1
            if counter % interp_freq == 0:
                counter = 0
                post_propagate_path.append(post_state.copy())
    return np.array(post_propagate_path)

def subsample_path(data, gt, c2g, csf, c, transit_pair_data, th=0):
    # print(c)
    subs_data, subs_transit_pair_data, subs_gt, subs_c2g, subs_csf, subs_c = [], [], [], [], [], []
    for j in range(len(data)):
        ### reset everything
        curr_cost = 0
        ### start from j-th waypoint
        current_data = data[j]
        for i in range(j, len(data)):
            curr_cost += c[i]
            if curr_cost > th and i > j:
                curr_cost = 0
                subs_data.append(current_data)
                current_data = data[i]
                # subs_transit_pair_data.append(transit_pair_data[i])
                subs_gt.append(gt[i-1])
                # subs_c2g.append(c2g[i])
                # subs_csf.append(csf[i])
                # subs_c.append(c[i])
    return [np.array(d) for d in [subs_data, subs_gt, c2g, csf, c, transit_pair_data]]#[subs_data, subs_gt, subs_c2g, subs_csf, subs_c, subs_transit_pair_data]]



def path_to_tensor_forward(env_id, 
                           path_dict,
                           normalize,
                           interpolate=False,
                           system="acrobot_obs",
                           th=0,
                           goal_aug=False):
    """
    [env_id, state, goal]
    """    
    if interpolate:
        path, costs_sofar, costs2go = interpolate_path(path_dict)
    else:
        path = path_dict['path']
        costs = path_dict['cost']
        costs2go = costs.copy()
        costs_sofar = costs.copy()
        for i in range(costs.shape[0]):
            costs2go[i] = costs[i:].sum()
            costs_sofar[i] = costs[:i].sum()
#         print(path.shape, costs2go.shape)
    start_goal = path_dict['start_goal']
    n_nodes = path.shape[0]
    state_size = path.shape[1]
    data = []
    gt = []
    c2g = []
    csf = []
    c = []
    transit_pair_data = []

    if system == 'quadrotor_obs':
        for i in range(len(path)):
            if path[i, 6] < 0:
                path[i, 3:7] *= -1

    # start to first path node
#    data.append(np.concatenate(([env_id], start_goal[0], start_goal[-1])))
#    gt.append(path[0, :])
    for i_start in range(n_nodes-1):
        data.append(np.concatenate(([env_id], path[i_start, :], path[-1, :])))
        #data.append(np.concatenate(([env_id], path[i_start, :], start_goal[-1])))
        transit_pair_data.append(np.concatenate(([env_id], path[i_start, :], path[i_start+1, :])))
        gt.append(path[i_start+1, :])
        c2g.append(costs2go[i_start])
        csf.append(costs_sofar[i_start])
        c.append(costs[i_start])
        # if goal_aug:
        #     ## goal aug
        #     for i_goal in range(i_start+1, n_nodes):#[n_nodes-1]:#
        #         data.append(np.concatenate(([env_id], path[i_start, :], path[i_goal, :])))
        #         gt.append(path[i_start+1, :])
    ## last path node to goal
    # data.append(np.concatenate(([env_id], path[-1, :], start_goal[-1])))
    # gt.append(start_goal[-1])

    # data.append(np.concatenate(([env_id], path[-1, :], path[-1, :])))
    # gt.append(path[-1, :])
    #c2g.append(0)

    '''
    subsample_path 
    '''
    if th > 0:
        data, gt, c2g, csf, c, transit_pair_data = \
            subsample_path(data, gt, c2g, csf, c, transit_pair_data, th=th)

    data = np.array(data)
    gt = np.array(gt)
    c2g = np.array(c2g)
    csf = np.array(csf)
    c = np.array(c)
    transit_pair_data = np.array(transit_pair_data)
            
    
    if normalize:
        if system == "acrobot_obs":
            data[:, [1,2,5,6]] /= np.pi
            data[:, [3,4,7,8]] /= 6
            gt[:, [0,1]] /= np.pi
            gt[:, [2,3]] /= 6
        elif system == "cartpole_obs":
            data[:, 1:] /= np.array([30, 40, np.pi, 2, 30, 40, np.pi, 2])
            gt /= np.array([30, 40, np.pi, 2])
        elif system == 'quadrotor_obs':
            data[:, 1:] /= np.array([# start
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
    
    return data, gt, c2g, csf, c, transit_pair_data



@click.command()
@click.option('--num', default=10)
@click.option('--system', default='quadrotor_obs')
@click.option('--traj_num', default=900)
@click.option('--setup', default='default_norm')
@click.option('--normalize', default=True)
@click.option('--interpolate', default=False)
@click.option('--subsample_th', default=-1.0)
@click.option('--goal_aug', default=False)
def main(num, system, traj_num, setup, normalize, interpolate, subsample_th, goal_aug):
    data, gt, cost_to_go, cost_so_far, cost = [], [], [], [], []
    transit_pair_data = []
    data_with_obs, c2g_with_obs = [], []
    for env_id in range(num):
        # infeasible_points = get_obs_states(env_id = env_id, system=system)
        for traj_id in tqdm(range(traj_num)):
            # try:
            path_dict = load_data(system, env_id, traj_id)
            data_lists = path_to_tensor_forward(env_id, 
                                                path_dict,
                                                normalize,
                                                interpolate=interpolate,
                                                system=system,
                                                th=subsample_th,
                                                goal_aug=goal_aug)
            d, g, c2g, csf, c, tp_d = data_lists
            data.append(d)
            gt.append(g)
            cost_to_go.append(c2g)
            cost_so_far.append(csf)
            cost.append(c)
            transit_pair_data.append(tp_d)
            # if traj_id % 50 == 0:
                # print(env_id, traj_id)
        # infeasible_points = get_obs_states(env_id = env_id, system=system)
        # print(infeasible_points.shape)
    data = np.concatenate(data, axis=0)
    gt = np.concatenate(gt, axis=0)
    cost_to_go = np.concatenate(cost_to_go, axis=0)
    cost_so_far = np.concatenate(cost_so_far, axis=0)
    cost = np.concatenate(cost, axis=0)
    transit_pair_data = np.concatenate(transit_pair_data, axis=0)

    print([d.shape for d in [data, gt, cost_to_go, cost_so_far, transit_pair_data]])

    Path("data/{}".format(setup)).mkdir(parents=True, exist_ok=True)
    np.save('data/{}/{}_path_data.npy'.format(setup, system), data)
    np.save('data/{}/{}_gt.npy'.format(setup, system), gt)
    np.save('data/{}/{}_cost_to_go.npy'.format(setup, system), cost_to_go)
    np.save('data/{}/{}_cost_so_far.npy'.format(setup, system), cost_so_far)
    np.save('data/{}/{}_cost.npy'.format(setup, system), cost)
    np.save('data/{}/{}_transit_pair_data.npy'.format(setup, system), transit_pair_data)




if __name__ == '__main__':
    main()
