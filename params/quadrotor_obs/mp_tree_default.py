import numpy as np

def get_params():
    params = {
        'solver_type': "cem_cuda",
        'n_problem': 128,  # 128,
        'n_sample': 32,  # 32,
        'n_elite': 4,  # 4,
        'n_t': 2,
        'max_it': 3,
        'converge_r': 1e-10,

        'dt': 2e-3,

        'mu_u': np.array([-10., 0., 0., 0.]),
        'sigma_u': np.array([15., 1., 1., 1.]),

        'mu_t': .4,
        'sigma_t': .4,
        't_max': .8,
        'verbose': False,  # True,#
        'step_size': 0.8,

        "goal_radius": 2.0,

        "sst_delta_near": .1,
        "sst_delta_drain": 0.01,
        "goal_bias": 0.02,

        "width": 1,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 1,
        "mpnet_weight_path": "/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/quadrotor_obs/mpnet-tree-batch-128.pt",
        "cost_predictor_weight_path": "",
        "cost_to_go_predictor_weight_path": "",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:0",

        "cost_reselection": False,
        "number_of_iterations": 1000,
        "weights_array": np.ones(13),
        'max_planning_time': 50,
    }

    return params
