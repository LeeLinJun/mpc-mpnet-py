import numpy as np

def get_params():
    params = {
        'n_problem': 1,
        'n_sample': 32,
        'n_elite': 4,
        'n_t': 1,
        'max_it': 5,
        'converge_r': 1e-10,

        'dt': 2e-3,

        'mu_u': [-10, 0, 0, 0],
        'sigma_u': [15, 1, 1, 1],

        'mu_t': 0.25,
        'sigma_t': 0.25,
        't_max': 1,

        'verbose':  False, #True,#
        'step_size': 1,

        "goal_radius": 2,

        "sst_delta_near": .1,
        "sst_delta_drain": 0.,
        "goal_bias": 0.0,

        "width": 1,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 5,
        "mpnet_weight_path": "mpnet/exported/output/quadrotor_obs/mpnet.pt",
        "cost_predictor_weight_path": "",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/quadrotor_obs/cost_to_go.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0.,
        "refine_threshold": 0.,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 10000,
        "weights_array": np.ones(13),
        "shm_max_steps": 100,

        'max_planning_time': 500,
    }

    return params

