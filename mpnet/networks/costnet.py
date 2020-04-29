from .voxel_encoder import VoxelEncoder
from .pnet_shallow import PNet

import torch
from torch import nn

class CostNet(nn.Module):
    def __init__(self, ae_input_size=32, ae_output_size=64,
                 in_channels=1,
                 state_size=4,
                 encoder = None,
                 control_size=0):
        super(CostNet, self).__init__()
        if encoder == None:
            self.encoder = VoxelEncoder(input_size=ae_input_size, 
                                    output_size=ae_output_size,
                                    in_channels=in_channels)
        else:
            self.encoder = encoder
        self.mlp = PNet(input_size=ae_output_size+state_size * 2,
                         output_size=1)
    
    def forward(self, x, obs):
        if obs is not None:
            z = self.encoder(obs)
            z_x = torch.cat((z,x), 1)
        else:
            z_x = x
        return self.mlp(z_x)
