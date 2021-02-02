import numpy as np

def get_params():
    params = {
        'solver_type': "cem",
        'n_problem': 1,
        'n_sample': 32,
        'n_elite': 2,
        'n_t': 1,
        'max_it': 5,
        'converge_r': 1e-3,

        'dt': 2e-3,

        'mu_u': [0],
        'sigma_u': [400],

        'mu_t': 0.4,
        'sigma_t': 0.5,
        't_max': 1,

        'verbose': False,  # True,#
        'step_size': 1,

        "goal_radius": 1.5,

        "sst_delta_near": .6,
        "sst_delta_drain": .3,
        "goal_bias": 0.05,

        "width": 4,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 5,
        "mpnet_weight_path": "mpnet/exported/output/cartpole_obs/mpnet_10k_external_small_model.pt",
        "cost_predictor_weight_path": "",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/cartpole_obs/cost_to_go_10k.pt"

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0.0,
        "refine_threshold": 0.0,
        "device_id": "cuda:0",

        "cost_reselection": False,
        "number_of_iterations": 40000,
        "weights_array": [1, 0.5, 1, .5],
        'max_planning_time': 50,
        "shm_max_steps": 40
    }
    return params
