import torch
# import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from .logger import Logger
import numpy as np
# import click
from tqdm import tqdm
from pathlib import Path
# import importlib

def train_network(network, data_loaders, network_name="mpnet",
                  lr=3e-4, epochs=1000, batch=128,
                  system_env="sst_envs", system="acrobot_obs", setup="default_norm",
                  using_step_lr=True, step_size=100, gamma=0.9,
                  loss_type="l1_loss", weight_save_epochs=25,
                  aug=False):
    train_loader, test_loader = data_loaders
    env_vox = torch.from_numpy(np.load('{}/data/{}_env_vox.npy'.format(system_env, system))).float()
    if torch.cuda.is_available():
        network = network.cuda()
        env_vox = env_vox.cuda()

    # optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    # optimizer = torch.optim.Adagrad(network.parameters(), lr=lr)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    if using_step_lr:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    logger = Logger("output/{}/{}/{}/".format(system, setup, network_name))

    def get_loss(output, label): 
        return eval("torch.nn.functional."+loss_type)(output, label)

    # loss_coeff = np.array([15, 20., 15.7, 2.])
    # loss_coeff /= np.sum(loss_coeff)

    with tqdm(range(epochs+1), total=epochs+1) as pbar:
        for ep in range(epochs+1):
            train_loss_list = []
            network.train()
            for data, label in train_loader:
                if aug:
                    data, label = network.aug(data, label)
                # prepare data
                inputs = data[:, 1:]
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    label = label.cuda()
                envs = env_vox[(data[:, 0]).long()].cuda()
                # execute
                optimizer.zero_grad()
                output = network(inputs, envs)
                loss_tensor = torch.zeros(label.size(1))
                for dim_i in range(label.size(1)):
                    loss_tensor[dim_i] = get_loss(output[:, dim_i], label[:, dim_i])  # * loss_coeff[dim_i]
                loss = torch.sum(loss_tensor)
                loss.backward()
                train_loss_list.append(loss_tensor.detach().cpu().numpy())
                optimizer.step()
            train_loss_list = np.array(train_loss_list)
            logger.train_step(train_loss_list.reshape(-1).mean(), ep, train_loss_list.mean(axis=0))
            if using_step_lr:
                scheduler.step()
            network.eval()
            for data, label in test_loader:
                eval_loss_list = []
                with torch.no_grad():
                    if torch.cuda.is_available():
                        data = data.cuda()
                        label = label.cuda()
                    inputs = data[:, 1:]
                    envs = env_vox[(data[:, 0]).long()]
                    output = network(inputs, envs)
                    loss_tensor = torch.zeros(label.size(1))
                    for dim_i in range(label.size(1)):
                        loss_tensor[dim_i] = get_loss(output[:, dim_i], label[:, dim_i])  # * loss_coeff[dim_i]
                    loss = torch.sum(loss_tensor)
                    eval_loss_list.append(loss_tensor.detach().cpu().numpy())
            eval_loss_list = np.array(eval_loss_list)
            logger.eval_step(eval_loss_list.mean(), ep)

            pbar.set_postfix({'eval_loss': '{0:1.5f}'.format(eval_loss_list.reshape(-1).mean()),
                              'train_loss': '{0:1.5f}'.format(train_loss_list.reshape(-1).mean()), })

            if ep % weight_save_epochs == 0:
                Path("output/{}/{}/{}".format(system, setup, network_name)).mkdir(parents=True, exist_ok=True)
                torch.save(network.state_dict(), "output/{}/{}/{}/ep{}.pth".format(system, setup, network_name, ep))
            pbar.update(1)
