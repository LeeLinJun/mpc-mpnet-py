source activate linjun
python data_generation.py --env_name quadrotor_obs  --N 1 --NP 2 --s 0 --sp 0 \
--max_iter 500000 --path_folder trajectories/quadrotor_obs/ \
--obs_file trajectories/quadrotor_obs/obs.pkl --obc_file trajectories/car_obs/obc.pkl \
--max_time=30
# 10x1000
