import numpy as np

def get_params():
    params = {
        'n_sample': 128,
        'n_elite': 8,
        'n_t': 5,
        'max_it': 10,
        'converge_r': 1e-1,

        'dt': 0.02,
        'mu_u': 0,
        'sigma_u': 4,
        'mu_t': 1.,
        'sigma_t': 0.4,
        't_max': 0.5,
        'verbose': False,
        'step_size': 0.1,

        "goal_radius": 2,
        "sst_delta_near": 0.5,
        "sst_delta_drain": 0.1,
        "width": 6,
        "hybrid": False,

    }

    return params
    
