import torch
import torch.utils.data as data_utils
import numpy as np

def np_to_loader(data, label, ind, batch, shuffle):
    th_data = torch.from_numpy(data[ind,:]).float()
    th_label = torch.from_numpy(label[ind, :]).float()
    dataset = data_utils.TensorDataset(th_data, th_label)
    return data_utils.DataLoader(dataset, batch_size=batch, shuffle=shuffle)

def get_loader(system, model, batch_size=128, setup='default'):
    path_data = np.load('{system}/{setup}/{model}_path_data.npy'.format(system=system, setup=setup, model=model))
    gt = np.load('{system}/{setup}/{model}_gt.npy'.format(system=system, setup=setup, model=model))
    # env_vox = np.load('env_vox.npy')
    # env_data = env_vox[path_data[:, 0].astype(int)]
    shuffle_ind = np.arange(path_data.shape[0])
    np.random.shuffle(shuffle_ind)
    np.random.seed(42)
    n_train = int(path_data.shape[0]*0.9)
    train_ind = shuffle_ind[:n_train]
    test_ind = shuffle_ind[n_train:]
    train_loader = np_to_loader(path_data, gt, train_ind, batch_size, True)
    test_loader = np_to_loader(path_data, gt, test_ind, batch_size, False)
    return train_loader, test_loader

def get_loader_cost(system, model, batch_size=128, setup='default'):
    path_data = np.load('{system}/{setup}/{model}_path_data.npy'.format(system=system, setup=setup, model=model))
    gt = np.load('{system}/{setup}/{model}_costs.npy'.format(system=system, setup=setup, model=model))
    gt = np.expand_dims(gt, axis=1)
    # env_vox = np.load('env_vox.npy')
    # env_data = env_vox[path_data[:, 0].astype(int)]
    shuffle_ind = np.arange(path_data.shape[0])
    np.random.shuffle(shuffle_ind)
    np.random.seed(42)
    n_train = int(path_data.shape[0]*0.9)
    train_ind = shuffle_ind[:n_train]
    test_ind = shuffle_ind[n_train:]
    train_loader = np_to_loader(path_data, gt, train_ind, batch_size, True)
    test_loader = np_to_loader(path_data, gt, test_ind, batch_size, False)
    return train_loader, test_loader
if __name__ == '__main__':
    train_loader, test_loader = get_loader()