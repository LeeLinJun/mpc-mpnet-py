source activate linjun
python benchmarks/benchmark.py --system quadrotor_obs --experiment_type mp_tree  --traj_id_offset 800 --num_traj 200 $@