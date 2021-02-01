source activate linjun
# python benchmarks/benchmark.py --system acrobot_obs --num_env 2 --traj_id_offset 0 --experiment_type shm_unseen  --traj_id_offset 0 --num_traj 100 $@
python benchmarks/benchmark.py --system acrobot_obs --num_env 1 --traj_id_offset 0 --experiment_type shm_unseen  --traj_id_offset 0 --num_traj 100 --env_id 2 --config default122 $@