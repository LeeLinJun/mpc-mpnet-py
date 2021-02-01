import numpy as np

def voxelize(start_goal, origin, sample_vox_res=0.2):
        # voxelize:
        return ((start_goal + origin)/sample_vox_res).astype(int)


import pickle
from pathlib import Path
from time import sleep
import shutil
import fcntl

def env_fn(env, model="quadrotor_obs", generalize=False):
    return "./trajectories/{model}_generalize/{env}/env{env}.pkl".format(model=model, env=env) if generalize else "./trajectories/{model}/{env}/env{env}.pkl".format(model=model, env=env)

def load_occ_table(env, table=set(), model="quadrotor_obs"):
    file_name = env_fn(env)
    try:
        with open(file_name, 'rb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            t = pickle.load(f)
        table = table.union(t)
        print('dict loaded, size:{}'.format(len(table)))
    except (FileNotFoundError):
        print('Created new occupancy table {}'.format(file_name))
        Path("./trajectories/{}/{}".format(model, env)).mkdir(parents=True, exist_ok=True)
        table = set()
        with open(file_name, 'wb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            pickle.dump(table, f)
    return table

def save_occ_table(env, table):
    file_name = env_fn(env)
    try:
        shutil.copyfile(file_name, file_name+'.bk')
        print('dict backed up')
        with open(file_name, 'wb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            pickle.dump(table, f)
    except:
        print('save dict exception')
        sleep(1)

def save_data(env, file_name, file_type, id_list, call_back, mode='wb', model="quadrotor_obs"):
    Path("./trajectories/{}/{}/{}".format(model, env, file_name)).mkdir(parents=True, exist_ok=True)
    # try:
    with open("trajectories/{model}/{env}/{fn}/{fn}_{id}.{type}".format(
                                                    model=model,
                                                    env=env, 
                                                    fn=file_name,
                                                    id="_".join(map(str, id_list.tolist())),
                                                    type=file_type), mode) as text_file:
        fcntl.flock(text_file.fileno(), fcntl.LOCK_EX)
        call_back(text_file)
        
    # except:
    #     print('save data exception')
    #     sleep(1)
