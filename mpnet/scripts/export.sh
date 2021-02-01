#! /bin/bash
source activate linjun
# python exported/export_mpnet.py --system acrobot_obs
python exported/export_cost_to_go.py --system acrobot_obs --setup d_n_itp --ep 400 --outputfn c2g_itp.pt
# python exported/export_cost_to_go.py --system acrobot_obs
# python exported/export_cost_so_far.py --system acrobot_obs
# python exported/export_cost.py --system acrobot_obs

# python exported/export_mpnet.py --system cartpole_obs 
#python exported/export_mpnet.py --system cartpole_obs  --ep 1000
#python exported/export_mpnet_branch.py --system cartpole_obs  --ep 1000

# python exported/export_mpnet.py --system cartpole_obs --setup default --outputfn mpnet_10k_nonorm.pt
# python exported/export_mpnet.py --system cartpole_obs --setup default_norm_aug --outputfn mpnet_10k_aug.pt --ep 500


# python exported/export_mpnet_external.py --system cartpole_obs --outputfn mpnet_10k_external.pt
# python exported/export_mpnet_external_v2.py --system cartpole_obs --outputfn mpnet_10k_external_v2_deep.pt
#python exported/export_mpnet_external_v2.py --system cartpole_obs --outputfn mpnet_10k_external_v2_deep.pt
# python exported/export_mpnet_external_v3_multigoal.py --system cartpole_obs --outputfn mpnet_10k_external_v3_multigoal.pt

### best model
# python exported/export_mpnet_external_small_model.py --system cartpole_obs --outputfn mpnet_10k_external_small_model.pt
###

#python exported/export_mpnet_pos_vel_external.py --system cartpole_obs --setup default --outputfn mpnet_pos_vel_external.pt
#python exported/export_mpnet_pos_vel_external_small_model.py --system cartpole_obs --setup default --outputfn mpnet_pos_vel_external.pt


# python exported/export_mpnet.py --system cartpole_obs --setup default_norm_subsample0.5  --outputfn mpnet_subsample0.5_10k.pt 


# python exported/export_mpnet.py --system cartpole_obs --setup default_norm_subsample0.2 --outputfn mpnet_subsample0.2_10k.pt 

# python exported/export_cost_to_go.py --system cartpole_obs --ep 2500

# python exported/export_cost_so_far.py --system cartpole_obs
# python exported/export_cost.py --system cartpole_obs  --ep 2500


# python exported/export_cost_to_go.py --system cartpole_obs --ep 1000 --network_type cost_to_go_obs --outputfn cost_to_go_obs
# python exported/export_cost_to_go.py --system cartpole_obs --ep 1000 --network_type cost_to_go_aug --outputfn cost_to_go_aug


# python exported/export_mpnet.py --system quadrotor_obs --ep 75 --outputfn mpnet_b1.pt --batch_size 1
# python exported/export_mpnet.py --system quadrotor_obs --ep 75 --outputfn mpnet.pt --batch_size 5
# python exported/export_cost_to_go.py --system quadrotor_obs  --ep 1000
# python exported/export_mpnet.py --system quadrotor_obs --ep 75 --outputfn mpnet-tree-batch-128.pt --batch_size 128


# python exported/export_mpnet.py --system quadrotor_obs --ep 1000 --outputfn mpnet_1k_l1_adagrad.pt --network_name mpnet_l1_adagrad

# python exported/export_mpnet_external_small_model_car_step_500.py --system car_obs --ep 950 --outputfn mpnet_10k_external_small_model_step_500.pt

# python exported/export_cost_to_go.py --system car_obs  --ep 1000
