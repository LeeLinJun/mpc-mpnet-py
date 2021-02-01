import numpy as np
import pickle
from obs_2d import pcd_to_voxel2d
from obs_3d import pcd_to_voxel3d
from utils import num_unseen_envs


def filepath(system, env_id, filetype):
    return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{system}/{filetype}_{env_id}.pkl".format(system=system, env_id=env_id, filetype=filetype)


def loader(system, env_id, filetype):
    return pickle.load(open(filepath(system, env_id, filetype), "rb"))


def process_and_save(system, obc_list, voxel_size=[32, 32], reshape_size=[-1, 1, 32, 32], dim=2, unseen=False):
    pcd_to_voxel = pcd_to_voxel2d if dim == 2 else pcd_to_voxel3d
    obs_vox = pcd_to_voxel(obc_list, voxel_size=voxel_size).reshape(reshape_size)
    print("Saving {}_env_vox{}.npy with dim {}".format(
        system, "_unseen" if unseen else "", obs_vox.shape))
    np.save("data/{}_env_vox{}.npy".format(system, "_unseen" if unseen else ""), obs_vox)


def gen2d(system):
    if system == "acrobot_obs":
        system_path = "acrobot_obs_backup_corrected"
    else:
        system_path = system
    # 10 envs
    process_and_save(system=system,
                     obc_list=np.array([loader(system_path, env_id, "obc").reshape(-1, 2) for env_id in range(10)]),
                     voxel_size=[32, 32],
                     reshape_size=[-1, 1, 32, 32],
                     unseen=False)
    # unseen 2 envs
    process_and_save(system=system,
                     obc_list=np.array([loader(system_path, env_id+10, "obc").reshape(-1, 2) for env_id in range(
                         3 if system == 'acrobot_obs' else num_unseen_envs)]),
                     voxel_size=[32, 32],
                     reshape_size=[-1, 1, 32, 32],
                     unseen=True)


def gen3d(system):
    system_path = system
    # 10 envs
    process_and_save(system=system_path,
                     obc_list=np.array([loader(system_path, env_id, "obc").reshape(-1, 3) for env_id in range(10)]),
                     voxel_size=[32, 32, 32],
                     reshape_size=[-1, 32, 32, 32],
                     dim=3,
                     unseen=False)

    # unseen 2 envs
    process_and_save(system=system_path,
                     obc_list=np.array([loader(system_path, env_id+10, "obc").reshape(-1, 3) for env_id in range(num_unseen_envs)]),
                     voxel_size=[32, 32, 32],
                     reshape_size=[-1, 32, 32, 32],
                     dim=3,
                     unseen=True)


def main():
    # for 2d envs
    for system in ['acrobot_obs', 'cartpole_obs', 'car_obs']:
        gen2d(system)
    # for 3d envs
    for system in ['quadrotor_obs']:
        gen3d(system)


if __name__ == '__main__':
    main()
