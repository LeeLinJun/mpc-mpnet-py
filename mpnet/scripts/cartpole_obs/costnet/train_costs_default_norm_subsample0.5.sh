#! /bin/bash
source activate linjun
python train_costs.py --system cartpole_obs --epochs 10000  --setup default_norm_subsample0.5