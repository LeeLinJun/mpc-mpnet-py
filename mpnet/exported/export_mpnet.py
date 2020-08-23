import torch
import numpy as np
import click

from export import export
# from networks.mpnet import MPNet

# from networks.mpnet_cartpole_obs import MPNet
from dataset.dataset import get_loader
from config import *
from pathlib import Path

@click.command()
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=10000)
@click.option('--outputfn', default="mpnet_10k.pt")
@click.option('--network_name', default='mpnet')
@click.option('--batch_size', default=1)
def main(system_env, system, setup, ep, outputfn, network_name, batch_size):
    if system == 'cartpole_obs':
        from networks.mpnet_cartpole_obs import MPNet
    elif system == 'acrobot_obs':
        from networks.mpnet_acrobot_obs import MPNet
    elif system == 'quadrotor_obs':
        from networks.mpnet_quadrotor_obs import MPNet
    else:
        print("Unrecognized model name")
        raise
    mpnet = MPNet(
        ae_input_size=32, 
        ae_output_size=output_size[system], 
        in_channels=in_channel[system], 
        state_size=state_size[system]).cuda()
    mpnet.load_state_dict(torch.load('output/{}/{}/{}/ep{}.pth'.format(system, setup, network_name, ep)))
    mpnet.train()
    # mpnet.eval()
    Path("exported/output/{}".format(system)).mkdir(exist_ok=True)

    export(mpnet, setup=setup, system_env=system_env, system=system, exported_path="exported/output/{}/{}".format(system, outputfn), batch_size=batch_size)

if __name__ == '__main__':
    main()
