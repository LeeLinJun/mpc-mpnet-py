import numpy as np

def get_params():
    params = {
        'solver_type' : "cem",
        'n_problem' : 1,
        'n_sample': 32,
        'n_elite': 4,
        'n_t': 2,
        'max_it': 3,
        'converge_r': 1e-5,
        
        'dt': 2e-3,

        'mu_u': np.array([1.0, 0]),
        'sigma_u': np.array([1.0, 0.5]),

        'mu_t': 0.5,
        'sigma_t': 0.5,
        't_max': 2,

        'verbose': False,#True,# 
        'step_size': 0.8,

        "goal_radius": 2.0,

        "sst_delta_near": .5,
        "sst_delta_drain": .2,
        "goal_bias": 0.08,

        "width": 8,
        "hybrid": False,
        "hybrid_p": 0.0,

        "cost_samples": 1,
        "mpnet_weight_path":"/media/arclabdl1/HD1/YLmiao/mpc-mpnet-cuda-yinglong/mpnet/exported/output/car_obs/mpnet_10k_external_small_model_step_500.pt",
        

        "cost_predictor_weight_path": "",
        "cost_to_go_predictor_weight_path": "",
        "refine": False,
        "using_one_step_cost": False,
        "refine_lr": 0,
        "refine_threshold": 0,
        "device_id": "cuda:3",

        "cost_reselection": False,
        "number_of_iterations": 2000,
        "weights_array": [1., 1., 1.],

        "shm_max_steps": 50,
        "max_planning_time": 50
    }

    return params
    
