source activate linjun
python3 data_generation_gen.py --env_name quadrotor_obs  --N $1 --NP 100 \
--max_iter 5000000 --path_folder trajectories/quadrotor_obs_generalize/ \
--obs_file trajectories/quadrotor_obs_generalize/obs.pkl --obc_file trajectories/quadrotor_obs_generalize/obc.pkl \
--max_time=500
# 10x1000
