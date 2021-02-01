import numpy as np
import sys
import pickle
import click


def pcd_to_voxel3d(points, voxel_size=(32, 32, 32), padding_size=(32, 32, 32)):
    voxels = [voxelize3d(points[i], voxel_size, padding_size) for i in range(len(points))]
    # return size: BxV*V*V
    return np.array(voxels)

def voxelize3d(points, voxel_size=(32, 32, 32), padding_size=(32, 32, 32), resolution=0.05):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.
    Args:
    `points`: pointcloud in 3D numpy.ndarray (shape: N * 3)
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters
    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    # calculate resolution based on boundary
    if abs(resolution) < sys.float_info.epsilon:
        print('error input, resolution should not be zero')
        return None, None

    """
    here the point cloud is centerized, and each dimension uses a different resolution
    """
    OCCUPIED = 1
    FREE = 0
    resolution = [(points[:,i].max() - points[:,i].min()) / voxel_size[i] for i in range(3)]
    resolution = np.array(resolution)
    #resolution = np.max(res)
    # remove all non-numeric elements of the said array
    points = points[np.logical_not(np.isnan(points).any(axis=1))]

    # filter outside voxel_box by using passthrough filter
    # TODO Origin, better use centroid?
    origin = (np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    points[:, 2] -= origin[2]

    # logical condition index
    x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution[0]), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution[1]), (points[:, 1] >= 0))
    z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution[2]), (points[:, 2] >= 0))
    # xy_logical = np.logical_and(x_logical, y_logical)
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    inside_box_points = points[xyz_logical]
    # inside_box_points = points[xy_logical]
    # init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
    voxels = np.zeros(padding_size)
    # centerlize to padding box
    center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2
    # TODO currently just use the binary hit grid
    x_idx = (center_points[:, 0] / resolution[0]).astype(int)
    y_idx = (center_points[:, 1] / resolution[1]).astype(int)
    z_idx = (center_points[:, 2] / resolution[2]).astype(int)
    voxels[x_idx, y_idx, z_idx] = OCCUPIED
    # voxels[x_idx, y_idx] = OCCUPIED
    return voxels
    #return voxels, inside_box_points


@click.command()
@click.option('--system', default='quadrotor_obs')
def main(system="acrobot_obs"):
    print("Processing {} env".format(system))
    system_path = system
    def filepath (system, env_id, filetype):
        # return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{system}/{filetype}_{env_id}.pkl".format(system=system_path, env_id=env_id, filetype=filetype)
        return "/media/arclabdl1/HD1/YLmiao/data/kinodynamic/quadrotor_obs/{filetype}_{env_id}.pkl".format(env_id=env_id, filetype=filetype)
    def loader(system, env_id, filetype): 
        return pickle.load(open(filepath(system_path, env_id, filetype), "rb"))

    obs_list = []
    #obs_list = [loader(system, env_id, "obs") for env_id in range(10)]
    # for env_id in range(10):
    #     print(env_id)
    #     print(loader(system_path, env_id, "obc").reshape(-1, 3))

    obc_list = np.array([loader(system_path, env_id, "obc").reshape(-1, 3) for env_id in range(10)])
    obs_vox = pcd_to_voxel3d(obc_list, voxel_size=[32, 32, 32]).reshape(-1, 32, 32, 32)
    print("Saving {}_env_vox.npy with dim {}".format(system, obs_vox.shape))
    np.save("data/{}_env_vox.npy".format(system), obs_vox)

if __name__ == '__main__':
    main()

