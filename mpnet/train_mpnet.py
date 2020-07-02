from dataset.dataset import get_loader
import importlib

import torch

import numpy as np
import click

from training_utils.trainer import train_network

@click.command()
@click.option('--ae_output_size', default=32, help='ae_output_size')
@click.option('--state_size', default=4, help='')
@click.option('--lr', default=3e-4, help='learning_rate')
@click.option('--epochs', default=1000, help='epochs')
@click.option('--batch', default=128, help='batch')
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--loss_type', default='l1_loss')
@click.option('--lr_step_size', default=100)
@click.option('--add_noise', default=True)
@click.option('--network_name', default="mpnet")
def main(ae_output_size, state_size, lr, epochs, batch, 
    system_env, system, setup, loss_type, lr_step_size, add_noise, network_name):
    # mpnet_module = importlib.import_module('.mpnet_{}'.format(system), package=".networks")
    try:
        if system == 'cartpole_obs':
            if network_name == 'mpnet':
                from networks.mpnet_cartpole_obs import MPNet
            elif network_name == 'mpnet_branch':
                from networks.mpnet_cartpole_obs_branch import MPNet

        elif system == 'acrobot_obs':
            from networks.mpnet_acrobot_obs import MPNet
    except:
        print("Unrecognized model name")
        raise

    mpnet = MPNet(ae_input_size=32, ae_output_size=ae_output_size, in_channels=1, state_size=4)

    data_loaders = get_loader(system_env, system, batch_size=batch, setup=setup)

    train_network(network=mpnet, data_loaders=data_loaders, network_name=network_name,
        lr=lr, epochs=epochs, batch=batch, 
        system_env=system_env, system=system, setup=setup,
        using_step_lr=True, step_size=lr_step_size, gamma=0.9,
        loss_type=loss_type, weight_save_epochs=25, add_noise=add_noise)

if __name__ == '__main__':
    main()
