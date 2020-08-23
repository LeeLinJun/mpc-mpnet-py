import torch
import numpy as np
import click
from torch import nn

from export import export
# from networks.mpnet import MPNet

# from networks.mpnet_cartpole_obs import MPNet
#from dataset.dataset import get_loader
# from config import *
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(),
                     nn.Linear(64, output_size))
    def forward(self, x):
        out = self.fc(x)
        return out



class Encoder(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    # adapted from SingleView 2
    def __init__(self, input_size=32, output_size=32):
            super(Encoder, self).__init__()
            input_size = [input_size, input_size]
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=7, kernel_size=[6,6], stride=[2,2]),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),
        )
            x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
            first_fc_in_features = 1
            for n in x.size()[1:]:
                first_fc_in_features *= n
            print('length of the output of one encoder')
            print(first_fc_in_features)
            self.head = nn.Sequential(
                nn.Linear(first_fc_in_features, 64),
                nn.PReLU(),
                nn.Linear(64, output_size)
            )
    def forward(self, x):
        # x shape: BxCxWxH
        #size = x.size()
        #x1 = x.permute(0, 1, 4, 2, 3).reshape(size[0], -1, size[2], size[3])# transpose to Bx(CxD)xWxH

        #x1, x2, x3 = self.encoder1(x1),self.encoder2(x2),self.encoder3(x3)
        #x1, x2, x3 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # cat x1 x2 x3 into x
        #x = torch.cat([x1, x2, x3], dim=1)
        x = self.head(x)
        return x
    
class KMPNet(nn.Module):
    def __init__(self, total_input_size, AE_input_size, mlp_input_size, output_size, CAE, MLP, loss_f):
        super(KMPNet, self).__init__()
        if CAE is None:
            self.encoder = None
        else:
            self.encoder = Encoder(AE_input_size, mlp_input_size-total_input_size)
        self.mlp = MLP(mlp_input_size, output_size)
        self.loss_f = loss_f
        self.opt = torch.optim.Adagrad(list(self.encoder.parameters())+list(self.mlp.parameters()))
        self.total_input_size = total_input_size
        self.AE_input_size = AE_input_size

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr, momentum=momentum)

    def forward(self, x, obs):
        # xobs is the input to encoder
        # x is the input to mlp
        if obs is not None:
            z = self.encoder(obs)
            mlp_in = torch.cat((z,x), 1)
        else:
            mlp_in = x
        res = self.mlp(mlp_in)
        #mean = 0.
        #stddev = 0.1
        #noise = torch.randn(res.size()) * stddev + mean
        #noise = noise.cuda()
        #return res + noise
        return res


def load_func(net, fname):
    if torch.cuda.is_available():
        #checkpoint = torch.load(fname, map_location='cuda:%d' % (torch.cuda.current_device()))
        checkpoint = torch.load(fname, map_location='cuda:0')
    else:
        checkpoint = torch.load(fname, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])

@click.command()
@click.option('--system_env', default='sst_envs')
@click.option('--system', default='car_obs')
@click.option('--setup', default='default_norm')
@click.option('--ep', default=10000)
@click.option('--outputfn', default="mpnet_10k_external_small_model_step_500.pt")
def main(system_env, system, setup, ep, outputfn):

    # mpnet = MPNet(
    #     ae_input_size=32, 
    #     ae_output_size=1024, 
    #     in_channels=1, 
    #     state_size=state_size[system]).cuda()
    mpnet = KMPNet(total_input_size=6, AE_input_size=32, mlp_input_size=38, output_size=3, CAE=Encoder, MLP=MLP, loss_f=None)
    mpnet.cuda()
    # mpnet.load_state_dict(torch.load('/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/cartpole_obs_lr0.001000_SGD_step_100/kmpnet_epoch_9950_direction_0_step_100.pkl'.format(system, setup, ep)))
    # load_func(mpnet, '/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/cartpole_obs_lr0.001000_SGD_step_100/kmpnet_epoch_9950_direction_0_step_100.pkl')
    # load_func(mpnet, '/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/cartpole_obs_3_lr0.001000_Adagrad_step_200/kmpnet_epoch_2600_direction_0_step_200.pkl')
    load_func(mpnet, '/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/car_obs_lr0.010000_Adagrad_step_500/kmpnet_epoch_950_direction_0_step_500.pkl')
    #load_func(mpnet, '/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/car_obs_lr0.010000_Adagrad_loss_mse_decoupled_step_200/kmpnet_epoch_400_direction_0_step_200.pkl')
    mpnet.train()
    #mpnet.eval()
    export(mpnet, setup=setup, system_env=system_env, system=system, exported_path="exported/output/{}/{}".format(system, outputfn))

if __name__ == '__main__':
    main()
