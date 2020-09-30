source activate linjun
python benchmarks/benchmark.py --system quadrotor_obs --experiment_type shm  --traj_id_offset 900 --num_traj 100 --config shmdefault_nc $@
