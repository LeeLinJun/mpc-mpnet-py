import numpy as np

def get_params(final_goal):
    n_t = 5
    state_dim = 4
    control_dim = 1
    weights = np.ones(state_dim)*1
    weights[2:] = 0.25
    n_sample = 1024
    n_elite = 32
    t = 1e-1
    dt = 2e-2

    mu_t, sigma_t = 1e-1, 4e-1
    t_min, t_max = 0, 5e-1

    mu_u = np.zeros((n_t*control_dim))
    sigma_u_diag = np.ones(n_t*control_dim)
    sigma_u_diag[:] = 4
    sigma_u = np.diag(sigma_u_diag)
    params = {
        'n_sample': n_sample,
        'n_elite': n_elite,
        'n_t': n_t,
        'weights': weights,
        'mu_u': mu_u,
        'sigma_u': sigma_u,
        't': t,
        'dt': dt,

        'mu_t': np.ones(n_t) * mu_t,
        'sigma_t': np.identity(n_t)*sigma_t,
        't_min': t_min,
        't_max': t_max,

        'state_dim': state_dim,
        'control_dim': control_dim,
        'converge_radius': 1e-2,
        'drop_radius': 1,
        'goal_radius': 10, #np.sqrt(2),
        'max_it': 20,
        'rolling_count': n_t,
        'bk_it': 2,
        'final_goal': final_goal,
        'mpc_mode': 'solve',#'mpc_mode': 'rolling'
        'max_plan_it': 300,


#         'planning_mode': 'line_search',
        'planning_mode': 'tree',
        'delta_near': 0.2,
        'delta_drain': 0.2,
    }

    return params
    