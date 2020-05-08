import torch
import numpy as np
import click

from export import export

from networks.mpnet import MPNet
from dataset.dataset import get_loader

@click.command()
@click.option('--system', default='sst_envs')
@click.option('--model', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=5000)
def main(system, model, setup, ep):
    state_size = {"acrobot_obs": 4}
    mpnet = MPNet(
        ae_input_size=32, 
        ae_output_size=1024, 
        in_channels=1, 
        state_size=state_size[model]).cuda()
    mpnet.load_state_dict(torch.load('output/{}/{}/ep{}.pth'.format(model, setup, ep)))
    mpnet.train()
    export(mpnet, setup=setup, system=system, model=model, exported_path="exported/output/mpnet5000.pt")

    
    # for env_id in range(10):
    #     i_th_env = env_vox[env_id].unsqueeze(0)
    #     # print(i_th_env.shape)
    #     torch.save(i_th_env, 'cpp/output/env_vox_{}.pt'.format())

if __name__ == '__main__':
    main()