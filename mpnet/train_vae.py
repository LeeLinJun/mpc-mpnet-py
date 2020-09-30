from dataset.dataset import get_loader
from networks.vae_mpnet import VAEMPNet

import torch

import numpy as np
import click

from training_utils.trainer import train_network

@click.command()
@click.option('--ae_output_size', default=1024, help='ae_output_size')
@click.option('--state_size', default=4, help='')
@click.option('--lr', default=5e-4, help='learning_rate')
@click.option('--epochs', default=10000, help='epochs')
@click.option('--batch', default=128, help='batch')
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--loss_type', default='smooth_l1_loss')
def main(ae_output_size, state_size, lr, epochs, batch, 
    system_env, system, setup, loss_type):
    mpnet = VAEMPNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4, code_size=32)

    data_loaders = get_loader(system_env, system, batch_size=batch, setup=setup)

    train_network(network=mpnet, data_loaders=data_loaders, network_name="mpnet_vae",
        lr=lr, epochs=epochs, batch=batch, 
        system_env=system_env, system=system, setup=setup,
        using_step_lr=True, step_size=50, gamma=0.9,
        loss_type=loss_type, weight_save_epochs=50)

if __name__ == '__main__':
    main()
