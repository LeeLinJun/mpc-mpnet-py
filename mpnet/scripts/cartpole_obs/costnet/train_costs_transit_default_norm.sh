#! /bin/bash
source activate linjun
python train_costs_transit.py --system cartpole_obs --epochs 10000  --setup default_norm