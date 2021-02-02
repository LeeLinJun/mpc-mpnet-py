source activate linjun
python3 data_generation.py --env_name quadrotor_obs  --N $1 --NP 32 \
--max_iter 5000000 --path_folder trajectories/quadrotor_obs/ \
--obs_file trajectories/quadrotor_obs/obs.pkl --obc_file trajectories/car_obs/obc.pkl \
--max_time=500
# 10x1000
