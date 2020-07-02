import numpy as np

def get_params():
    params = {
        'n_sample': 32,
        'n_elite': 4,
        'n_t': 5,
        'max_it': 5,
        'converge_r': 5e-1,

        'mu_u': 0,
        'sigma_u': 300,
        'dt': 2e-3,

        'mu_t': 0.1,
        'sigma_t': 0.1,
        't_max': 0.5,

        'verbose': True,#False,#True,# 
        'step_size': 1,

        "goal_radius": 1.5,

        "sst_delta_near": 0.6,
        "sst_delta_drain": 0.1,
        "goal_bias": 0.2,

        "width": 4,
        "hybrid": False,
        "hybrid_p": 0,

        "cost_samples": 1,
        "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

        "cost_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_10k.pt",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 1,
        
        
    }

    return params
    
