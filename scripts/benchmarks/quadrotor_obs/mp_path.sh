source activate linjun
python benchmarks/benchmark.py --system quadrotor_obs --experiment_type mp_path  --traj_id_offset 900 --num_traj 100 $@
