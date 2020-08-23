import numpy as np

def get_params():
    params = {
            "integration_step": 2e-3,
            "random_seed": 0,
            "goal_radius": 2,
            "sst_delta_near": 0.5,
            "sst_delta_drain": 0.1,
            "width": 1,        
            "min_time_steps": 50,
            "max_time_steps": 500,
            "number_of_iterations": 600000,

            "max_planning_time": 500
            }

    return params


