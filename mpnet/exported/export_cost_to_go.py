import torch
import numpy as np
import click

from export import export
from networks.mpnet import MPNet
from networks.costnet import CostNet 
from config import *

@click.command()
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=10000)
def main(system_env, system, setup, ep):
    mpnet = MPNet(
        ae_input_size=32, 
        ae_output_size=1024, 
        in_channels=1, 
        state_size=state_size[system]).cuda()
    costnet = CostNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4, encoder=mpnet.encoder).cuda()
    costnet.load_state_dict(torch.load('output/{}/{}/cost_to_go/ep{}.pth'.format(system, setup, ep)))
    costnet.eval()
    export(costnet, setup=setup, system_env=system_env, system=system, exported_path="exported/output/{}/cost_to_go_10k.pt".format(system))
if __name__ == '__main__':
    main()