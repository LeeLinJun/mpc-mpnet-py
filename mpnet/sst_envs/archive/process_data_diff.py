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

    for k in range(len(ref_control)):
        max_steps = int(np.round(ref_time[k]/step_size))
        #print(state, ref_path[k])
        state = ref_path[k].copy()
        waypoints[-1] = ref_path[k]
        for step in range(1, max_steps+1):
            state = dynamics.propagate(state.copy(), [ref_control[k]], 1, step_size).copy()
            if (step % interval_steps == 0) or (step == max_steps):
                waypoints.append(state.copy())
                #print(step, state)
    waypoints[-1] = ref_path[-1].copy()
    #print(waypoints[-1], ref_path[-1])
    waypoints = np.array(waypoints)
    #print(waypoints.shape)
    #print(waypoints) 
    return waypoints

def path_to_tensor_forward(env_id, path_dict, normalize):
    """
    [env_id, state, goal]
    """    
    path = interpolate_path(path_dict)
#     path = path_dict['path']
    start_goal = path_dict['start_goal']
    n_nodes = path.shape[0]
    state_size = path.shape[1]
    data =[] 
    gt = []
    # start to first path node
#    data.append(np.concatenate(([env_id], start_goal[0], start_goal[-1])))
#    gt.append(path[0, :])
    for i_start in range(n_nodes-1):
        data.append(np.concatenate(([env_id], path[i_start, :], start_goal[-1])))
        gt_diff = path[i_start+1, :] - path[i_start, :]
        if gt_diff[0] > np.pi:
            gt_diff[0] -= np.pi *2
        elif gt_diff[0] < -np.pi:
            gt_diff[0] -= np.pi *2
        if gt_diff[1] > np.pi:
            gt_diff[1] -= np.pi *2
        elif gt_diff[1] < -np.pi:
            gt_diff[1] -= np.pi *2    
        gt.append(gt_diff)
        ## goal aug
#         for i_goal in range(i_start+1, n_nodes):#[n_nodes-1]:#
#             data.append(np.concatenate(([env_id], path[i_start, :], path[i_goal, :])))
#             gt.append(path[i_start+1, :])
            
    # last path node to goal
    data.append(np.concatenate(([env_id], path[-1, :], start_goal[-1])))
    gt.append(start_goal[-1])
    data = np.array(data)
    gt = np.array(gt)
    if normalize:
        data[:, [1,2,5,6]] /= np.pi
        data[:, [3,4,7,8]] /= 6
        gt[:, [0,1]] /= np.pi
        gt[:, [2,3]] /= 6
    return data, gt


@click.command()
@click.option('--num', default=10)
@click.option('--system', default='acrobot_obs')
@click.option('--traj_num', default=1000)
@click.option('--setup', default='default')
@click.option('--normalize', default=True)
def main(num, system, traj_num, setup, normalize):
    data, gt = [], []
    for env_id in range(num):
        for traj_id  in range(traj_num):
            # try:
            path_dict = load_data(system, env_id, traj_id)
            d, g = path_to_tensor_forward(env_id, path_dict, normalize)
            data.append(d)
            gt.append(g)
            # except:
                # pass
            print(env_id, traj_id)
    data = np.concatenate(data, axis=0)
    gt = np.concatenate(gt, axis=0)
    print(data.shape,gt.shape)

    np.save('{}/{}_path_data.npy'.format(setup, system), data)
    np.save('{}/{}_gt.npy'.format(setup, system), gt)


if __name__ == '__main__':
    main()
