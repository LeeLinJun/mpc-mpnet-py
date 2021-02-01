#! /bin/bash
source activate linjun
python train_costs.py --system acrobot_obs --epochs 400  --setup d_n_aug_itp --batch 128 --lr=3e-4
