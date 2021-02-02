#! /bin/bash
source activate linjun
python train_costs.py --system cartpole_obs --epochs 200  --setup d_n_aug_itp --batch 128