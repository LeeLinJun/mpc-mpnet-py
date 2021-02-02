source activate linjun
python benchmarks/benchmark.py --system cartpole_obs --experiment_type mp_tree  --traj_id_offset 800 --num_traj 200 $@