#! /bin/bash
source activate linjun
python train_mpnet.py --system acrobot_obs --epochs 500  --setup d_n_aug_itp20