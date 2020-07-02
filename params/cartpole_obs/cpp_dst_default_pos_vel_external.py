import numpy as np

def get_params():
    params = {
        'n_sample': 32,
        'n_elite': 5,
        'n_t': 1,
        'max_it': 5,
        'converge_r': 1e-1,
        
        'dt': 2e-3,

        'mu_u': 0,
        'sigma_u': 300,

        'mu_t': 0.3,
        'sigma_t': 0.3,
        't_max': 1.0,

        'verbose': False,#True,# 
        'step_size': 1,

        "goal_radius": 1.5,

        "sst_delta_near": 0.6,
        "sst_delta_drain": 0.1,
        "goal_bias": 0.1,

        "width": 4,
        "hybrid": False,
        "hybrid_p": 0.,

        "cost_samples": 1,
        "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_pos_vel_external.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v2_deep.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",

        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

        "cost_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_10k.pt",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:2",

        "cost_reselection": False,
        "number_of_iterations": 10000,
        
        
    }

    return params
    
