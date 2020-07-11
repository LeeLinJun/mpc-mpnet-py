#! /bin/bash
source activate linjun
python train_mpnet.py --system cartpole_obs --epochs 10000 --lr_step_size 100 --setup default_subsample0.5