import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from .logger import Logger
import numpy as np
import click
from tqdm import tqdm
from pathlib import Path
import importlib

def train_network(network, data_loaders, network_name="mpnet",
    lr=1e-3, epochs=5000, batch=128, 
    system_env="sst_envs", system="acrobot_obs", setup="default_norm",
    using_step_lr=True, step_size=50, gamma=0.9,
    loss_type="l1_loss", weight_save_epochs=50):
    train_loader, test_loader = data_loaders
    env_vox = torch.from_numpy(np.load('{}/{}_env_vox.npy'.format(system_env, system))).float()
    if torch.cuda.is_available():
        network = network.cuda()
        env_vox = env_vox.cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    if using_step_lr:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    eval_loss = 0
    train_loss = 0
    logger = Logger("output/{}/{}/{}/".format(system, setup, network_name))
    with tqdm(range(epochs+1), total=epochs+1) as pbar:
        for i in range(epochs+1):
            train_loss_list = []
            network.train()
            for data, label in train_loader:
                ## prepare data
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                inputs = data[:,1:]
                envs = env_vox[(data[:, 0]).long()]
                ## execute
                optimizer.zero_grad()
                output = network(inputs, envs)
                loss = eval("torch.nn.functional."+loss_type)(output, label)
                loss.backward()
                train_loss_list.append(loss.item())
                optimizer.step()
            train_loss = np.mean(train_loss_list)
            
            logger.train_step(loss, i)
            if using_step_lr:
                scheduler.step(i)
            network.eval()
            for data, label in test_loader:
                eval_loss_list = []
                with torch.no_grad():
                    if torch.cuda.is_available():
                        data = data.cuda()
                        label = label.cuda()                
                    inputs = data[:,1:]
                    envs = env_vox[(data[:,0]).long()]
                    output = network(inputs, envs)
                    loss = eval("torch.nn.functional."+loss_type)(output, label)
                    eval_loss_list.append(loss.item())
            eval_loss = np.mean(eval_loss_list)
            pbar.set_postfix({'eval_loss': '{0:1.5f}'.format(eval_loss),
                              'train_loss': '{0:1.5f}'.format(train_loss),})
            logger.eval_step(eval_loss, i)

           
            if i % weight_save_epochs == 0:
                Path("output/{}/{}/{}".format(system, setup, network_name)).mkdir(parents=True, exist_ok=True)
                torch.save(network.state_dict(), "output/{}/{}/{}/ep{}.pth".format(system, setup, network_name, i))
            pbar.update(1)