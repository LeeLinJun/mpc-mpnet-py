#! /bin/bash
source activate linjun
python train_mpnet.py --system quadrotor_obs --epochs 200 --setup default_norm_ss1 --batch 1024 --lr 1e-3 $@
