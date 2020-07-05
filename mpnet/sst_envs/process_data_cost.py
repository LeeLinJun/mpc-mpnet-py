from utils import load_data
import numpy as np
import pickle
import re
import click
from systems.acrobot import Acrobot
from pathlib import Path

def get_dynamics(dynamics):
    sys_dict = {
        "acrobot_obs": "Acrobot",
        "cartpole_obs": "CartPole"
    }
    return eval("sys_dict[dynamics]()")

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

def subsample_path(data, gt, c2g, csf, c, transit_pair_data, th=0):
    curr_cost = 0
    # print(c)
    subs_data, subs_transit_pair_data, subs_gt, subs_c2g, subs_csf, subs_c = [], [], [], [], [], []
    for i in range(len(data)):
        curr_cost += c[i]
        # print(curr_cost)
        if curr_cost < th and i < len(data)-1 and i > 0:
            continue
        else:
            curr_cost = 0
            subs_data.append(data[i])
            subs_transit_pair_data.append(transit_pair_data[i])
            subs_gt.append(gt[i])
            subs_c2g.append(c2g[i])
            subs_csf.append(csf[i])
            subs_c.append(c[i])
    return [np.array(d) for d in [subs_data, subs_gt, subs_c2g, subs_csf, subs_c, subs_transit_pair_data]]



def path_to_tensor_forward(env_id, path_dict, normalize, interpolate=False, system="acrobot_obs", th=0, goal_aug=False):
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
    data =[] 
    gt = []
    c2g = []
    csf = []
    c = []
    transit_pair_data = []
    # start to first path node
#    data.append(np.concatenate(([env_id], start_goal[0], start_goal[-1])))
#    gt.append(path[0, :])
    for i_start in range(n_nodes-1):
        data.append(np.concatenate(([env_id], path[i_start, :], path[-1])))
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
            
    # last path node to goal
    #data.append(np.concatenate(([env_id], path[-1, :], start_goal[-1])))
    #gt.append(start_goal[-1])
    #c2g.append(0)
    data = np.array(data)
    gt = np.array(gt)
    c2g = np.array(c2g)
    csf = np.array(csf)
    c = np.array(c)
    transit_pair_data = np.array(transit_pair_data)

    if False: #th > 0:
        data, gt, c2g, csf, c, transit_pair_data = \
            subsample_path(data, gt, c2g, csf, c, transit_pair_data, th=th)

    if normalize:
        if system == "acrobot_obs":
            data[:, [1,2,5,6]] /= np.pi
            data[:, [3,4,7,8]] /= 6
            gt[:, [0,1]] /= np.pi
            gt[:, [2,3]] /= 6
        elif system == "cartpole_obs":
            data[:, 1] /= 30
            data[:, 2] /= 40
            data[:, 3] /= np.pi
            data[:, 4] /= 2
            data[:, 5] /= 30
            data[:, 6] /= 40
            data[:, 7] /= np.pi
            data[:, 8] /= 2

            gt[:, 0] /= 30
            gt[:, 1] /= 40
            gt[:, 2] /= np.pi
            gt[:, 3] /= 2
        else:
            raise NotImplementedError("unkown dynamics")
    return data, gt, c2g, csf, c, transit_pair_data


@click.command()
@click.option('--num', default=10)
@click.option('--system', default='cartpole_obs')
@click.option('--traj_num', default=2000)
@click.option('--setup', default='default_norm')
@click.option('--normalize', default=True)
@click.option('--interpolate', default=False)
@click.option('--subsample_th', default=.0)
@click.option('--goal_aug', default=False)
def main(num, system, traj_num, setup, normalize, interpolate, subsample_th, goal_aug):
    data, gt, cost_to_go, cost_so_far, cost = [], [], [], [], []
    transit_pair_data = []
    for env_id in range(num):
        for traj_id  in range(traj_num):
            # try:
            path_dict = load_data(system, env_id, traj_id)
            d, g, c2g, csf, c, tp_d = path_to_tensor_forward(env_id, 
                                                             path_dict,
                                                             normalize,
                                                             interpolate=interpolate,
                                                             system=system,
                                                             th=subsample_th,
                                                             goal_aug=goal_aug)
            data.append(d)
            gt.append(g)
            cost_to_go.append(c2g)
            cost_so_far.append(csf)
            cost.append(c)
            transit_pair_data.append(tp_d)
            if traj_id % 50 == 0:
                print(env_id, traj_id)
    data = np.concatenate(data, axis=0)
    gt = np.concatenate(gt, axis=0)
    cost_to_go = np.concatenate(cost_to_go, axis=0)
    cost_so_far = np.concatenate(cost_so_far, axis=0)
    cost = np.concatenate(cost, axis=0)
    transit_pair_data = np.concatenate(transit_pair_data, axis=0)
    print(data.shape, gt.shape, cost_to_go.shape, cost_so_far.shape, transit_pair_data.shape)

    Path("{}".format(setup)).mkdir(parents=True, exist_ok=True)
    np.save('{}/{}_path_data.npy'.format(setup, system), data)
    np.save('{}/{}_gt.npy'.format(setup, system), gt)
    np.save('{}/{}_cost_to_go.npy'.format(setup, system), cost_to_go)
    np.save('{}/{}_cost_so_far.npy'.format(setup, system), cost_so_far)
    np.save('{}/{}_cost.npy'.format(setup, system), cost)
    np.save('{}/{}_transit_pair_data.npy'.format(setup, system), transit_pair_data)



if __name__ == '__main__':
    main()
