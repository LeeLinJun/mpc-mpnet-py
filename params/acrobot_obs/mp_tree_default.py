def get_params():
    params = {
        'solver_type': "cem_cuda",
        'n_problem': 64,
        'n_sample': 32,
        'n_elite': 4,
        'n_t': 3,
        'max_it': 3,
        'converge_r': 1e-1,

        'dt': 2e-2,
        'mu_u': [0],
        'sigma_u': [4],
        'mu_t': 1e-1,
        'sigma_t': 5e-1,
        't_max': 0.5,

        'verbose': False,  # True,#
        'step_size': 0.8,

        "goal_radius": 2.0,

        "sst_delta_near": 1,
        "sst_delta_drain": 1e-1,
        "goal_bias": 0.08,

        "width": 6,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 1,
        "mpnet_weight_path": "mpnet/exported/output/acrobot_obs/mpnet_10k.pt",

        "cost_predictor_weight_path": "",
        "cost_to_go_predictor_weight_path": "",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 2000,
        "weights_array": [1, 1.0, .2, .2],
        'max_planning_time': 50,
    }

    return params
