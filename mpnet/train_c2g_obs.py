from dataset.dataset import get_loader_cost
from networks.costnet import CostNet
import torch

import numpy as np
import click

from training_utils.trainer import train_network
 
@click.command()
@click.option('--ae_output_size', default=1024, help='ae_output_size')
@click.option('--state_size', default=4, help='')
@click.option('--lr', default=1e-3, help='learning_rate')
@click.option('--epochs', default=10000, help='epochs')
@click.option('--batch', default=128, help='batch')
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--loss_type', default='l1_loss')
@click.option('--load_from', default='mpnet')
@click.option('--network_type', default='cost_to_go_obs')
@click.option('--data_type', default='data_with_obs')
@click.option('--label_type', default='c2g_with_obs')
@click.option('--from_exported', default=True)
def main(ae_output_size, state_size, lr, epochs, batch, 
    system_env, system, setup, loss_type, load_from, network_type,
    data_type, label_type, from_exported):
    if from_exported:
        import sys
        sys.path.append("/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported")
        from exported.export_mpnet_external_small_model import KMPNet, load_func, Encoder, MLP
        mpnet = KMPNet(total_input_size=8, AE_input_size=32, mlp_input_size=40, output_size=4, CAE=Encoder, MLP=MLP, loss_f=None)
        load_func(mpnet, '/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/cartpole_obs_4_lr0.010000_Adagrad_step_200/kmpnet_epoch_3150_direction_0_step_200.pkl')
        
        costnet = CostNet(ae_input_size=32, ae_output_size=32, in_channels=1, state_size=4, encoder=mpnet.encoder)
    else:
        from networks.mpnet import MPNet
        mpnet = MPNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4)
        mpnet.load_state_dict(torch.load('output/{}/{}/{}/ep10000.pth'.format(system, setup,load_from)))
    
        costnet = CostNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4, encoder=mpnet.encoder)
    # for param in costnet.encoder.parameters():
    #     param.requires_grad = False

    data_loaders = get_loader_cost(system_env, system, batch_size=batch, setup=setup, label_type=label_type, data_type=data_type)

    train_network(network=costnet, data_loaders=data_loaders, 
            network_name=network_type,
        lr=lr, epochs=epochs, batch=batch, 
        system_env=system_env, system=system, setup=setup,
        using_step_lr=True, step_size=100, gamma=0.9,
        loss_type=loss_type, weight_save_epochs=50)


if __name__ == "__main__":
    main()
