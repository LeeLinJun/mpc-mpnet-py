import torch
import numpy as np
import click

from export import export
from networks.mpnet import MPNet
from networks.costnet import CostNet 

@click.command()
@click.option('--system', default='sst_envs')
@click.option('--model', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=10000)
def main(system, model, setup, ep):
    state_size = {"acrobot_obs": 4}
    mpnet = MPNet(
        ae_input_size=32, 
        ae_output_size=1024, 
        in_channels=1, 
        state_size=state_size[model]).cuda()
    costnet = CostNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4, encoder=mpnet.encoder).cuda()
    costnet.load_state_dict(torch.load('output/{}/{}/cost_transit/ep{}.pth'.format(model, setup, ep)))
    costnet.eval()
    export(costnet, setup=setup, system=system, model=model, exported_path="exported/output/cost_10k.pt")
if __name__ == '__main__':
    main()