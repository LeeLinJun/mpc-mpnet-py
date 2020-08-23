#! /bin/bash
source activate linjun
python train_c2g_obs.py --system cartpole_obs --epochs 1000  --setup default_norm