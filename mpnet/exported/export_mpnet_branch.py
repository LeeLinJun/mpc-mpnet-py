import torch
import numpy as np
import click

from export import export
# from networks.mpnet import MPNet

# from networks.mpnet_cartpole_obs import MPNet
from dataset.dataset import get_loader
from config import *

@click.command()
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=10000)
@click.option('--outputfn', default="mpnet_10k_branch.pt")
def main(system_env, system, setup, ep, outputfn):
    if system == 'cartpole_obs':
        from networks.mpnet_cartpole_obs_branch import MPNet
    elif system == 'acrobot_obs':
        from networks.mpnet_acrobot_obs import MPNet
    else:
        print("Unrecognized model name")
        raise
    mpnet = MPNet(
        ae_input_size=32, 
        ae_output_size=32, 
        in_channels=1, 
        state_size=state_size[system]).cuda()
    mpnet.load_state_dict(torch.load('output/{}/{}/mpnet_branch/ep{}.pth'.format(system, setup, ep)))
    mpnet.train()
    # mpnet.eval()
    export(mpnet, setup=setup, system_env=system_env, system=system, exported_path="exported/output/{}/{}".format(system, outputfn))

if __name__ == '__main__':
    main()
