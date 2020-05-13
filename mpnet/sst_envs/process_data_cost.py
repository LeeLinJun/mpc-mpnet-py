from utils import load_data
import numpy as np
import pickle
import re
import click
from systems.acrobot import Acrobot

def interpolate_path(path_dict, dynamics=Acrobot(), interval_steps=20, step_size=2e-2):
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

def path_to_tensor_forward(env_id, path_dict, normalize, interpolate=True):
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
    # start to first path node
#    data.append(np.concatenate(([env_id], start_goal[0], start_goal[-1])))
#    gt.append(path[0, :])
    for i_start in range(n_nodes-1):
        data.append(np.concatenate(([env_id], path[i_start, :], start_goal[-1])))
        gt.append(path[i_start+1, :])
        c2g.append(costs2go[i_start])
        csf.append(costs_sofar[i_start])
        c.append(costs[i_start])
        ## goal aug
#         for i_goal in range(i_start+1, n_nodes):#[n_nodes-1]:#
#             data.append(np.concatenate(([env_id], path[i_start, :], path[i_goal, :])))
#             gt.append(path[i_start+1, :])
            
    # last path node to goal
    #data.append(np.concatenate(([env_id], path[-1, :], start_goal[-1])))
    #gt.append(start_goal[-1])
    #c2g.append(0)
    data = np.array(data)
    gt = np.array(gt)
    c2g = np.array(c2g)
    csf = np.array(csf)
    c = np.array(c)
    if normalize:
        data[:, [1,2,5,6]] /= np.pi
        data[:, [3,4,7,8]] /= 6
        gt[:, [0,1]] /= np.pi
        gt[:, [2,3]] /= 6
    return data, gt, c2g, csf, c


@click.command()
@click.option('--num', default=10)
@click.option('--system', default='acrobot_obs')
@click.option('--traj_num', default=1000)
@click.option('--setup', default='default_norm')
@click.option('--normalize', default=True)
@click.option('--interpolate', default=False)
def main(num, system, traj_num, setup, normalize, interpolate):
    data, gt, cost_to_go, cost_so_far, cost = [], [], [], [], []
    for env_id in range(num):
        for traj_id  in range(traj_num):
            # try:
            path_dict = load_data(system, env_id, traj_id)
            d, g, c2g, csf, c = path_to_tensor_forward(env_id, path_dict, normalize, interpolate=interpolate)
            data.append(d)
            gt.append(g)
            cost_to_go.append(c2g)
            cost_so_far.append(csf)
            cost.append(c)
            if traj_id % 50 == 0:
                print(env_id, traj_id)
    data = np.concatenate(data, axis=0)
    gt = np.concatenate(gt, axis=0)
    cost_to_go = np.concatenate(cost_to_go, axis=0)
    cost_so_far = np.concatenate(cost_so_far, axis=0)
    cost = np.concatenate(cost, axis=0)
    print(data.shape, gt.shape, cost_to_go.shape, cost_so_far.shape)

    np.save('{}/{}_path_data.npy'.format(setup, system), data)
    np.save('{}/{}_gt.npy'.format(setup, system), gt)
    np.save('{}/{}_cost_to_go.npy'.format(setup, system), cost_to_go)
    np.save('{}/{}_cost_so_far.npy'.format(setup, system), cost_so_far)
    np.save('{}/{}_cost.npy'.format(setup, system), cost)



if __name__ == '__main__':
    main()
