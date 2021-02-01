#! /bin/bash
source activate linjun
python train_costs.py --system acrobot_obs --epochs 1000  --setup d_n_itp --batch 512
