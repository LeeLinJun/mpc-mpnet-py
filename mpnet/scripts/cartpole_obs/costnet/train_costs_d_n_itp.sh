#! /bin/bash
source activate linjun
python train_costs.py --system cartpole_obs --epochs 1000  --setup d_n_itp