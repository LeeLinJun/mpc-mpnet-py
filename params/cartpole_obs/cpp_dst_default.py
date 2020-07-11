import numpy as np

def get_params():
    params = {
        
        'n_problem': 1,
        'n_sample': 32,
        'n_elite': 8,
        'n_t': 2,
        'max_it': 5,
        'converge_r': 0.1,

        'dt': 2e-3,

        'mu_u': 0,
        'sigma_u': 400,

        'mu_t': 0.1,
        'sigma_t': 0.2,
        't_max': 1,

        'verbose':  False,#True,#
        'step_size': 1,

        "goal_radius": 1.5,

        "sst_delta_near": .3,
        "sst_delta_drain": 0.1,
        "goal_bias": 0.05,

        "width": 4,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 1,
        "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_small_model.pt",
        #"mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_external_v2_deep.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k.pt",

        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_10k_nonorm.pt",
        # "mpnet_weight_path":"mpnet/exported/output/cartpole_obs/mpnet_subsample0.5_10k.pt",

        # "cost_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_10k.pt",
        "cost_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_10k.pt",

        "cost_to_go_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_to_go_obs.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 100000,
        "weights_array": [1, 1, 1, 0.5],
        'max_planning_time': 50,


    }

    return params