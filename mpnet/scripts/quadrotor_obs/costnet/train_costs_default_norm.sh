#! /bin/bash
source activate linjun
python train_costs.py --system quadrotor_obs --epochs 1000  --setup default_norm --state_size 13 --ae_output_size 64 --batch 1024