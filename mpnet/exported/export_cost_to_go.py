import torch
import numpy as np
import click

from export import export
from networks.costnet import CostNet 
from config import *
from pathlib import Path

@click.command()
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='acrobot_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=10000)
@click.option('--from_exported', default=False)
@click.option('--network_type', default="cost_to_go")
@click.option('--outputfn', default="cost_to_go.pt")
def main(system_env, system, setup, ep, from_exported, network_type, outputfn):
    if from_exported:
        from exported.export_mpnet_external_small_model import KMPNet, load_func, Encoder, MLP
        mpnet = KMPNet(total_input_size=8, AE_input_size=32, mlp_input_size=40, output_size=4, CAE=Encoder, MLP=MLP, loss_f=None).cuda()
        load_func(mpnet, '/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/cartpole_obs_4_lr0.010000_Adagrad_step_200/kmpnet_epoch_3150_direction_0_step_200.pkl')
        costnet = CostNet(ae_input_size=32, ae_output_size=32, in_channels=1, state_size=state_size[system], encoder=mpnet.encoder.cuda()).cuda()

    else:
        if system == 'quadrotor_obs':
            from networks.mpnet_quadrotor_obs import MPNet

        mpnet = MPNet(
            ae_input_size=32, 
            ae_output_size=output_size[system], 
            in_channels=in_channel[system], 
            state_size=state_size[system]).cuda()
        costnet = CostNet(ae_input_size=32, ae_output_size=output_size[system], in_channels=in_channel[system], state_size=state_size[system], encoder=mpnet.encoder).cuda()
    
    costnet.load_state_dict(torch.load('output/{}/{}/{}/ep{}.pth'.format(system, setup, network_type, ep)))
    costnet.eval()
    Path("exported/output/{}".format(system)).mkdir(exist_ok=True)

    export(costnet, setup=setup, system_env=system_env, system=system, exported_path="exported/output/{}/{}".format(system, outputfn))
if __name__ == '__main__':
    main()