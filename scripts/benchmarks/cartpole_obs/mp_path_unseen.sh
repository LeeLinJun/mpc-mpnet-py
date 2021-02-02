source activate linjun
python benchmarks/benchmark.py --system cartpole_obs --num_env 2 --traj_id_offset 0 --experiment_type mp_path_unseen  --traj_id_offset 0 --num_traj 100 $@