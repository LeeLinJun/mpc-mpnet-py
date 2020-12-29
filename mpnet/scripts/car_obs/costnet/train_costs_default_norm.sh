#! /bin/bash
source activate linjun
python train_costs.py --system car_obs --epochs 1000  --setup d_n_aug_itp --state_size 3 --ae_output_size 64 --batch 1024