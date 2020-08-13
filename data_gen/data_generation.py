"""
using SST* to generate near-optimal paths in specified environment
"""

import sys
sys.path.append('/home/srip19-pointcloud/linjun/robotics/mpc-mpnet-py/deps/sparse_rrt-1')
from sparse_rrt import _sst_module
from sparse_rrt.systems import standard_cpp_systems
# import random
# from sparse_rrt import _deep_smp_module

import argparse
# from sparse_rrt import _sst_module
# from sparse_rrt.systems import standard_cpp_systems
import numpy as np
import time
import pickle

import os
import gc
from multiprocessing import Process, Queue
import quadrotor_obs_gen
import quadrotor_sg_gen
from pcd_tools.voxel_dict import load_occ_table, save_occ_table

def main(args):
    if args.env_name == 'quadrotor_obs':
        params = {
            "integration_step": 2e-3,
            "random_seed": 0,
            "goal_radius": 2,
            "sst_delta_near": 0.5,
            "sst_delta_drain": 0.1,
            "width": 1,        
            "min_time_steps": 50,
            "max_time_steps": 500,
            "number_of_iterations": 600000}

        goal_radius = params['goal_radius']
        random_seed = params['random_seed']
        sst_delta_near = params['sst_delta_near']
        sst_delta_drain = params['sst_delta_drain']

        min_time_steps = params['min_time_steps']
        max_time_steps = params['max_time_steps']
        integration_step = params['integration_step']
        width = params['width']

        obs_list, obc_list = quadrotor_obs_gen.obs_gen(None, None, N_pc=40000, width=width)
        # for i in range(len(obs_list)):
        #     file = open(args.path_folder+'obs_%d.pkl' % (i+args.s), 'wb')
        #     pickle.dump(obs_list[i], file)
        #     file = open(args.path_folder+'obc_%d.pkl' % (i+args.s), 'wb')
        #     pickle.dump(obc_list[i], file)
        # exit(0)

    ###################################################################################

    def plan_one_path_sst(env, start, end, out_queue, path_file, control_file, cost_file, time_file, env_id=None, id_list=None):        
        planner = _sst_module.SSTWrapper(
            state_bounds=env.get_state_bounds(),
            control_bounds=env.get_control_bounds(),
            distance=env.distance_computer(),
            start_state=start,
            goal_state=end,
            goal_radius=goal_radius,
            random_seed=random_seed,
            sst_delta_near=sst_delta_near,
            sst_delta_drain=sst_delta_drain
        )
        # generate a path by using SST to plan for some maximal iterations
        time0 = time.time()
        #print('obs: %d, path: %d' % (i, j))
        for iter in range(args.max_iter):
            planner.step(env, min_time_steps, max_time_steps, integration_step)
           
            # if iter%5000 == 0:
            #     if planner.get_solution() is not None:
            #         print(time.time() - time0)
            if time.time() - time0 > args.max_time:
                break
        solution = planner.get_solution()
        plan_time = time.time() - time0
        if solution is None:
            out_queue.put(0)
        else:
            
            print('path succeeded.')
            path, controls, cost = solution
            print(path)
            path = np.array(path)
            controls = np.array(controls)
            cost = np.array(cost)

            file = open(path_file, 'wb')
            pickle.dump(path, file)
            file.close()
            file = open(control_file, 'wb')
            pickle.dump(controls, file)
            file.close()
            file = open(cost_file, 'wb')
            pickle.dump(cost, file)
            file.close()
            file = open(time_file, 'wb')
            pickle.dump(plan_time, file)
            file.close()

            assert env_id is not None
            assert id_list is not None

            saved = False
            while not saved:
                try:
                    table = load_occ_table(env_id)         
                    table.add(tuple(id_list))
                    save_occ_table(env_id, table)
                    saved = True
                except EOFError:
                    time.sleep(3)
                    #pass
            
            out_queue.put(1)
    ####################################################################################
    queue = Queue(1)
    # for i in range(args.N):
    for i in [args.N]:
        # load the obstacle by creating a new environment
        if args.env_name == 'quadrotor_obs':
            env_constr = standard_cpp_systems.RectangleObs3D
            env = env_constr(obs_list[i], width, 'quadrotor')

        # generate rec representation of obs
        obs_recs = []
        for k in range(len(obs_list[i])):
            # for each obs setting
            obs_recs.append([[obs_list[i][k][0]-width/2,obs_list[i][k][1]-width/2],
                             [obs_list[i][k][0]-width/2,obs_list[i][k][1]+width/2],
                             [obs_list[i][k][0]+width/2,obs_list[i][k][1]+width/2],
                             [obs_list[i][k][0]+width/2,obs_list[i][k][1]-width/2]])
        state_bounds = env.get_state_bounds()
        low = []
        high = []
        for j in range(len(state_bounds)):
            low.append(state_bounds[j][0])
            high.append(state_bounds[j][1])

        paths = []
        actions = []
        costs = []
        times = []
        suc_n = 0

        for j in range(args.NP):
            plan_start = time.time()
            while True:
                # randomly sample collision-free start and goal
                #start = np.random.uniform(low=low, high=high)
                #end = np.random.uniform(low=low, high=high)
                # set the velocity terms to zero
                if args.env_name == 'quadrotor_obs':
                    start, end, id_list = quadrotor_sg_gen.start_goal_gen(low, high, width, obs_list[i], obs_recs, table=load_occ_table(i))
                dir = args.path_folder+str(i+args.s)+'/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                # path_file = dir+args.path_file+'_%d'%(j+args.sp) + ".pkl"
                # control_file = dir+args.control_file+'_%d'%(j+args.sp) + ".pkl"
                # cost_file = dir+args.cost_file+'_%d'%(j+args.sp) + ".pkl"
                # time_file = dir+args.time_file+'_%d'%(j+args.sp) + ".pkl"
                # sg_file = dir+args.sg_file+'_%d'%(j+args.sp)+".pkl"

                path_id = "_".join(id_list.astype(str).tolist())
                path_file = dir+args.path_file+path_id+ ".pkl"
                control_file = dir+args.control_file + path_id + ".pkl"
                cost_file = dir+args.cost_file + path_id + ".pkl"
                time_file = dir+args.time_file + path_id + ".pkl"
                sg_file = dir+args.sg_file + path_id +".pkl"
                # p = Process(target=plan_one_path_sst, args=(env, start, end, queue, path_file, control_file, cost_file, time_file, i, id_list))
                # p.start()
                # p.join()
                plan_one_path_sst(env, start, end, queue, path_file, control_file, cost_file, time_file, i, id_list)
                res = queue.get()
                print('obtained result:')
                print(res)
                if res:
                    # plan successful
                    file = open(sg_file, 'wb')
                    sg = [start, end]
                    pickle.dump(sg, file)
                    file.close()
                    break
            print('path planning time: %f' % (time.time() - plan_start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='quadrotor_obs')
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--N_obs', type=int, default=6)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--NP', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=2000000)
    parser.add_argument('--path_folder', type=str, default='./trajectories/quadrotor_obs/')
    parser.add_argument('--path_file', type=str, default='path')
    parser.add_argument('--control_file', type=str, default='control')
    parser.add_argument('--cost_file', type=str, default='cost')
    parser.add_argument('--time_file', type=str, default='time')
    parser.add_argument('--sg_file', type=str, default='start_goal')
    parser.add_argument('--obs_file', type=str, default='./trajectories/quadrotor_obs/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./trajectories/quadrotor_obs/obc.pkl')
    parser.add_argument('--max_time', type=float, default=500.)
    args = parser.parse_args()
    main(args)
