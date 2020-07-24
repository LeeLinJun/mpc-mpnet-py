import pickle
import numpy as np

def load_data(model, env, traj_id):
    if model == 'acrobot_obs':
        model = 'acrobot_obs_backup'
    filepath = lambda var: "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{env}/{var}_{id}.pkl".format(model=model,env=env, var=var, id=traj_id)      
    load_pkl = lambda var: pickle.load(open(filepath(var), "rb"))
    keys = ["control", "path", "start_goal", "time", 'cost']
    return dict(zip(keys, [load_pkl(key) for key in keys]))
 
def acrobot_visualize_point(state):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    LENGTH = 20.
    x1 = LENGTH * np.cos(state[STATE_THETA_1] - np.pi / 2) 
    x2 = x1 + LENGTH * np.cos(state[STATE_THETA_1] + state[STATE_THETA_2] - np.pi/2)
    y1 = LENGTH * np.sin(state[STATE_THETA_1] - np.pi / 2) 
    y2 = y1 + LENGTH * np.sin(state[STATE_THETA_1] + state[STATE_THETA_2] - np.pi / 2)
    return x1, y1, x2, y2

def cartpole_visualize_point(state):
    H = 0.5
    L = 2.5
    x2 = state[0] + (L) * np.sin(state[2])
    y2 = -(L) * np.cos(state[2])
    return state[0], H, x2, y2

def get_obs(model, filetype):
    def filepath (model, env_id, filetype):
            return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{filetype}_{env_id}.pkl".format(model=model, env_id=env_id, filetype=filetype)
    def loader(model, env_id, filetype): 
        return pickle.load(open(filepath(model, env_id, filetype), "rb"))
    #obs_list = [loader(model, env_id, "obs") for env_id in range(10)]
    obs_list = np.array([loader(model, env_id, "obs").reshape(-1, 2) for env_id in range(10)])
    return obs_list
