#! /bin/bash
source activate linjun
python train_costs.py --system cartpole_obs --epochs 1000  --setup default_norm --network_type cost_to_go_aug  --aug True