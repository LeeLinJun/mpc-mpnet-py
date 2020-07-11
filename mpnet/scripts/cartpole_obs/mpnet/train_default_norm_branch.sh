#! /bin/bash
source activate linjun
python train_mpnet.py --system cartpole_obs --epochs 10000 --setup default_norm --batch 1024 --network_name mpnet_branch