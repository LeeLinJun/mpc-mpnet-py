#! /bin/bash
source activate linjun
python train_mpnet.py --system cartpole_obs --epochs 500 --lr_step_size 100 --setup default_norm_aug --loss_type l1_loss