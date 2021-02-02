from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pickle
import numpy as np

num_unseen_envs = 2

def load_data(model, env, traj_id):
    if model == 'acrobot_obs':
        # model = 'acrobot_obs_backup'
        model = 'acrobot_obs_backup_corrected'

    def filepath(var): return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{env}/{var}_{id}.pkl".format(
        model=model, env=env, var=var, id=traj_id)

    def load_pkl(var): return pickle.load(open(filepath(var), "rb"))
    keys = ["path", "start_goal", 'cost', 'control']
    return dict(zip(keys, [load_pkl(key) for key in keys]))


def load_data_unseen(model, env, traj_id):
    if model == 'acrobot_obs':
        # model = 'acrobot_obs_backup'
        model = 'acrobot_obs_backup_corrected'
    def filepath(var): return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{env}/{var}_{id}.pkl".format(
        model=model, env=env+10, var=var, id=traj_id)

    def load_pkl(var): return pickle.load(open(filepath(var), "rb"))
    keys = ["path", "start_goal", 'cost', 'control']
    return dict(zip(keys, [load_pkl(key) for key in keys]))

def get_obs(model, filetype='obs'):
    if model == 'acrobot_obs':
        model = 'acrobot_obs_backup'

    def filepath(model, env_id, filetype):
        return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{filetype}_{env_id}.pkl".format(model=model, env_id=env_id, filetype=filetype)

    def loader(model, env_id, filetype):
        return pickle.load(open(filepath(model, env_id, filetype), "rb"))
    #obs_list = [loader(model, env_id, "obs") for env_id in range(10)]
    obs_list = np.array([loader(model, env_id, "obs").reshape(-1, 2)
                         for env_id in range(10)])
    return obs_list


def get_obs_unseen(model, filetype='obs'):
    if model == 'acrobot_obs':
        model = 'acrobot_obs_backup'

    def filepath(model, env_id, filetype):
        return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{filetype}_{env_id}.pkl".format(model=model, env_id=env_id, filetype=filetype)

    def loader(model, env_id, filetype):
        return pickle.load(open(filepath(model, env_id, filetype), "rb"))
    #  obs_list = [loader(model, env_id, "obs") for env_id in range(10)]
    obs_list = np.array(
        [loader(model, env_id+10, "obs").reshape(-1, 2) for env_id in range(3 if model == 'acrobot_obs_backup' else num_unseen_envs)])
    return obs_list


def get_obs_3d(model='quadrotor_obs', filetype='obs'):
    def filepath(model, env_id, filetype):
        return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{filetype}_{env_id}.pkl".format(model=model, env_id=env_id, filetype=filetype)

    def loader(model, env_id, filetype):
        return pickle.load(open(filepath(model, env_id, filetype), "rb"))
    #obs_list = [loader(model, env_id, "obs") for env_id in range(10)]
    obs_list = np.array([loader(model, env_id, "obs").reshape(-1, 3)
                         for env_id in range(10)])
    return obs_list


def get_obs_3d_unseen(model='quadrotor_obs', filetype='obs'):
    def filepath(model, env_id, filetype):
        return "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{filetype}_{env_id}.pkl".format(model=model, env_id=env_id, filetype=filetype)

    def loader(model, env_id, filetype):
        return pickle.load(open(filepath(model, env_id, filetype), "rb"))
    #obs_list = [loader(model, env_id, "obs") for env_id in range(10)]
    obs_list = np.array(
        [loader(model, env_id+10, "obs").reshape(-1, 3) for env_id in range(num_unseen_envs)])
    return obs_list


def acrobot_visualize_point(state):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    LENGTH = 20.
    x1 = LENGTH * np.cos(state[STATE_THETA_1] - np.pi / 2)
    x2 = x1 + LENGTH * \
        np.cos(state[STATE_THETA_1] + state[STATE_THETA_2] - np.pi/2)
    y1 = LENGTH * np.sin(state[STATE_THETA_1] - np.pi / 2)
    y2 = y1 + LENGTH * \
        np.sin(state[STATE_THETA_1] + state[STATE_THETA_2] - np.pi / 2)
    return x1, y1, x2, y2


def cartpole_visualize_point(state):
    H = 0.5
    L = 2.5
    x2 = state[0] + (L) * np.sin(state[2])
    y2 = -(L) * np.cos(state[2])
    return state[0], H, x2, y2

# def quadrotor_obs_bbox()


def draw_line_3d(ax, p, p_index, color='b', alpha=1):
    for p_i in p_index:
        ax.plot3D(p[p_i, 0], p[p_i, 1], p[p_i, 2], c=color, alpha=alpha)


def centered_box_to_points_3d(center, size):
    half_size = [s/2 for s in size]
    direction, p = [1, -1], []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([center[di] + [x_d, y_d, z_d][di] * half_size[0]
                          for di in range(3)])
    return p


def rot_frame_3d(state, frame_size=0.25):
    b, c, d, a = state[3:7]
    rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                        [2 * b * c - 2 * a * d, 2 * a**2 - 1 +
                            2 * c**2, 2 * c * d + 2 * a * b],
                        [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
    quadrotor_frame = np.array([[frame_size, 0, 0],
                                [0, frame_size, 0],
                                [-frame_size, 0, 0],
                                [0, -frame_size, 0]]).T
    quadrotor_frame = rot_mat @ quadrotor_frame + state[:3].reshape(-1, 1)
    return quadrotor_frame


def q_to_points_3d(state):
    quadrotor_frame = rot_frame_3d(state)
    max_min, direction = [np.max(quadrotor_frame, axis=1), np.min(
        quadrotor_frame, axis=1)], [1, 0]
    p = []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([max_min[x_d][0], max_min[y_d][1], max_min[z_d][2]])
    return np.array(p)


def draw_box_3d(ax, p, color='b', alpha=1, surface_color='blue', linewidths=1, edgecolors='k'):
    index_lists = [[[0, 4], [4, 6], [6, 2], [2, 0], [0, 1], [1, 5], [5, 7], [7, 3], [3, 1], [1, 5]],
                   [[4, 5]],
                   [[6, 7]],
                   [[2, 3]]]
    for p_i in index_lists:
        draw_line_3d(ax, np.array(p), p_i, color=color, alpha=alpha)
    edges = [[p[e_i] for e_i in f_i] for f_i in [[0, 1, 5, 4],
                                                 [4, 5, 7, 6],
                                                 [6, 7, 3, 2],
                                                 [2, 0, 1, 3],
                                                 [2, 0, 4, 6],
                                                 [3, 1, 5, 7]]]
    faces = Poly3DCollection(
        edges, linewidths=linewidths, edgecolors=edgecolors)
    faces.set_facecolor(surface_color)
    faces.set_alpha(0.1)
    ax.add_collection3d(faces)


def visualize_quadrotor_path(path, start_state, goal_state, obs_list, draw_bbox=True, width=1, savefig=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    counter = 0

    for obs in obs_list:
        draw_box_3d(ax, centered_box_to_points_3d(center=obs, size=[width]*3))

    ax.scatter(start_state[0], start_state[1], start_state[2], c='red')
    ax.scatter(goal_state[0], goal_state[1], goal_state[2], c='orange')
    draw_box_3d(ax, q_to_points_3d(start_state), alpha=0.3,
                surface_color="orange", linewidths=0.)
    draw_box_3d(ax, q_to_points_3d(goal_state), alpha=0.3,
                surface_color="orange", linewidths=0.)

    if path is not None:
        ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='blue')
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='blue')

        for waypoint in path:
            f = rot_frame_3d(waypoint)
            ax.scatter(f[0], f[1], f[2], color='red', s=10)
            ax.plot(f[0, [0, 2]], f[1, [0, 2]], f[2, [0, 2]], c='b')
            ax.plot(f[0, [1, 3]], f[1, [1, 3]], f[2, [1, 3]], c='b')

            if draw_bbox:
                draw_box_3d(ax, q_to_points_3d(waypoint), alpha=0.3,
                            surface_color="orange", linewidths=0.)

            ax.set_xlim3d(-5, 5)
            ax.set_ylim3d(-5, 5)
            ax.set_zlim3d(-5, 5)
            if savefig:
                fig.savefig("figs/{}.png".format(counter))
            counter += 1

    return fig, ax
