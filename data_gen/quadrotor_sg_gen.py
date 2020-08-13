import numpy as np
import pickle
from pcd_tools.voxel_dict import voxelize

def valid_state(state, obs_lists, width=1., radius=0.25):
    def centered_box_to_points_3d(center, size):
        half_size = [s/2 for s in size]
        direction, p = [1, -1], []
        for x_d in direction:
            for y_d in direction:
                for z_d in direction:
                    p.append([center[di] + [x_d, y_d, z_d][di] * half_size[0] for di in range(3)])
        return p

    def rot_frame_3d(state, frame_size=0.25):
        b, c, d, a = state[3:7]
        rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                            [2 * b * c - 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d + 2 * a * b],
                            [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
        quadrotor_frame = np.array([[frame_size, 0, 0],
                                    [0, frame_size, 0],
                                    [-frame_size, 0, 0],
                                    [0, -frame_size, 0]]).T
        quadrotor_frame = rot_mat @ quadrotor_frame + state[:3].reshape(-1, 1)
        return quadrotor_frame

    def q_to_points_3d(state):
        quadrotor_frame = rot_frame_3d(state)   
        max_min, direction = [np.max(quadrotor_frame, axis=1), np.min(quadrotor_frame, axis=1)], [1, 0]
        p = []
        for x_d in direction:
            for y_d in direction:
                for z_d in direction:
                    p.append([max_min[x_d][0], max_min[y_d][1], max_min[z_d][2]])
        return np.array(p)

    for obs in obs_lists:
        corners = centered_box_to_points_3d(center=obs, size=[width]*3)
        obs_min_max = [np.min(corners, axis=0), np.max(corners, axis=0)]
        quadrotor_frame = rot_frame_3d(state, radius)   
        quadrotor_min_max = [np.min(quadrotor_frame, axis=1), np.max(quadrotor_frame, axis=1)]
#         print(quadrotor_min_max, obs_min_max)
        if quadrotor_min_max[0][0] <= obs_min_max[1][0] and quadrotor_min_max[1][0] >= obs_min_max[0][0] and\
            quadrotor_min_max[0][1] <= obs_min_max[1][1] and quadrotor_min_max[1][1] >= obs_min_max[0][1] and\
            quadrotor_min_max[0][2] <= obs_min_max[1][2] and quadrotor_min_max[1][2] >= obs_min_max[0][2]:
                return False
    return True


def start_goal_gen(low, high, width, obs_list, obs_recs, min_distance=3, table=set()):
    # using obs information and bound, to generate good start and goal
    while True:
        start = np.random.uniform(low=low, high=high)
        start[3:] = 0
        start[6] = 1
        end = np.random.uniform(low=low, high=high)
        end[3:] = 0
        end[6] = 1
        id_list = voxelize(np.concatenate((start[:3], end[:3])), 5)
        if np.linalg.norm(start[:2] - end[:2]) <= min_distance or tuple(id_list) in table:  # 30
            continue
        if valid_state(start, obs_list, width=width) and valid_state(end, obs_list, width=width):
            print(id_list)
            break
    return start, end, id_list
