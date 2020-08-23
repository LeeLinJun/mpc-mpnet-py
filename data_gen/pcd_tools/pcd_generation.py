'''
This implements conversion from obstacle list to point cloud list.
'''
import numpy as np

def rectangle_pcd(obs_list, width, num_sample=1400):
    '''
    :obs_list: a list of (N,2) obstacle midpoint locations
    :width: width of the obstacle
    '''
    rec_list = []
    pcd_list = []
    # convert obs to rectangle (low, high)
    for i in range(len(obs_list)):
        obs = obs_list[i]  # (N,2)
        # low: obs_list - width / 2
        # high: obs_list + width / 2
        low = obs - width / 2
        high = obs + width / 2
        # generate
        pcds = []
        for n in range(num_sample):
            pcd = np.random.uniform(low=low, high=high).reshape(-1, 1, 2)
            pcds.append(pcd)
        # concatenate: l x (N*1*2) => (N*k*2)
        pcds = np.concatenate(pcds, axis=1)
        pcd_list.append(pcds)
    pcd_list = np.array(pcd_list)
    return pcd_list


def rectangle_pcd_3d(obs_list, width, num_sample=2000):
    '''
    :obs_list: a list of (N,2) obstacle midpoint locations
    :width: width of the obstacle
    '''
    rec_list = []
    pcd_list = []
    # convert obs to rectangle (low, high)
    for i in range(len(obs_list)):
        obs = obs_list[i]  # (N, 3)
        low = obs - width / 2
        high = obs + width / 2
        # generate
        pcds = []
        for n in range(num_sample):
            pcd = np.random.uniform(low=low, high=high).reshape(-1, 1, 3)
            pcds.append(pcd)
        # concatenate: l x (N*1*2) => (N*k*2)
        pcds = np.concatenate(pcds, axis=1)
        pcd_list.append(pcds)
    pcd_list = np.array(pcd_list)
    return pcd_list