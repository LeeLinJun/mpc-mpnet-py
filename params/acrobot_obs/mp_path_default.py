import numpy as np


def get_params():
    params = {
        'solver_type': "cem",
        'n_problem': 1,
        'n_sample': 64,
        'n_elite': 4,
        'n_t': 1,
        'max_it': 3,
        'converge_r': 1e-1,

        'dt': 2e-2,
        'mu_u': [0],
        'sigma_u': [6],
        'mu_t': 2e-1,
        'sigma_t': 0.4,
        't_max': 2,
        'verbose': False,  # True, #
        'step_size': 0.75,

        "goal_radius": 2,
        "sst_delta_near": 0.4,
        "sst_delta_drain": 2e-1,
        "goal_bias": 0.05,

        "width": 6,
        "hybrid": True,
        "hybrid_p": 0.3,
        "cost_samples": 5,
        "mpnet_weight_path": "mpnet/exported/output/acrobot_obs/mpnet_10k.pt",

        "cost_predictor_weight_path": "",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/acrobot_obs/c2g_itp.pt",


        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 40000,
        "weights_array": [1, 1, .2, .2],
        'max_planning_time': 50,
        'shm_max_steps': 1
    }

    return params
