import numpy as np

def get_params():
    params = {
        'n_sample': 32,
        'n_elite': 4,
        'n_t': 5,
        'max_it': 5,
        'converge_r': 1e-1,

        'dt': 2e-2,
        'mu_u': 0,
        'sigma_u': 4,
        'mu_t': 5e-2,
        'sigma_t': 0.5,
        't_max': 1,
        'verbose': False,
        'step_size': 0.75,

        "goal_radius": 2,
        "sst_delta_near": 1,
        "sst_delta_drain": 1e-1,
        "width": 6,
        "hybrid": False,

    }

    return params
    
