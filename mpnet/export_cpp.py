import torch
import numpy as np
import click

from networks.mpnet import MPNet, MPNetExported
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
    mpnet.load_state_dict(torch.load('output/{}/{}/ep{}'.format(model, setup, ep)))
    mpnet.train()
    # mpnet.eval()
    # mpnet_exported = MPNetExported(
    #     ae_input_size=32, 
    #     ae_output_size=1024, 
    #     in_channels=1, 
    #     state_size=state_size[model])
    # mpnet_exported.encoder = mpnet.encoder
    # mpnet_exported.pnet = mpnet.pnet
    # mpnet_exported.train()

    env_vox = torch.from_numpy(np.load('{}/{}_env_vox.npy'.format(system, model))).float()
    train_loader, test_loader = get_loader(system, model, batch_size=1, setup=setup)
    
    env_input = env_vox[train_loader.dataset[0:1][0][0:1, 0].long()].cuda()
    state_input = train_loader.dataset[0:1][0][0:1, 1:].cuda()
    
    output = mpnet(state_input, env_input)
    print(output)
    traced_script_module = torch.jit.trace(mpnet, (state_input, env_input))

    serilized_module = torch.jit.script(mpnet)
    serilized_output = serilized_module(state_input, env_input)
    print(serilized_output)
    traced_script_module.save("cpp/output/mpnet5000.pt")
    
    for env_id in range(10):
        i_th_env = env_vox[env_id].unsqueeze(0)
        # print(i_th_env.shape)
        torch.save(i_th_env, 'cpp/output/env_vox_{}.pt'.format())
if __name__ == '__main__':
    main()