import numpy as np

def get_params():
    params = {
        'solver_type' : "cem_cuda",
        'n_problem' : 8,
        'n_sample': 64,
        'n_elite': 2,
        'n_t': 1,
        'max_it': 8,
        'converge_r': 1e-1,
        
        'dt': 2e-3,

        'mu_u': 0,
        'sigma_u': 400,

        'mu_t': 0.4,
        'sigma_t': 0.5,
        't_max': 0.6,

        'verbose': False,#True,# 
        'step_size': 0.8,

        "goal_radius": 1.5,

        "sst_delta_near": .2,
        "sst_delta_drain": .1,
        "goal_bias": 0.1,

        "width": 4,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 1,
        "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_small_model.pt",
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
        "weights_array": [1, .5, 1, .5],

        
    }

    return params
    
