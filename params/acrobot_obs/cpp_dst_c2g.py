import numpy as np

def get_params():
    # params from cpp_dst_gr2_n1_d0_1_s32_e4_step3.py
    params = {
        'n_sample': 32,
        'n_elite': 4,
        'n_t': 3,
        'max_it': 5,
        'converge_r': 1e-1,

        'dt': 2e-2,
        'mu_u': 0,
        'sigma_u': 4,
        'mu_t': 1e-1,
        'sigma_t': 5e-1,
        't_max': 0.5,
        'verbose': False,
        'step_size': 0.75,

        "goal_radius": 2,
        "sst_delta_near": 1,
        "sst_delta_drain": 1e-1,
        "width": 6,
        "hybrid": False,
        "hybrid_p": 0.,
        "cost_samples": 10,
        "mpnet_weight_path":"mpnet/exported/output/mpnet_10k.pt",
    
        "cost_predictor_weight_path": "mpnet/exported/output/cost_10k.pt",
        "cost_to_go_predictor_weight_path": "mpnet/exported/output/cost_to_go_10k.pt",

        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
    }

    return params
    
