from .voxel_encoder_smaller import VoxelEncoder
from .pnet_shallow import PNet

import torch
from torch import nn

class MPNet(nn.Module):
    def __init__(self, ae_input_size=32, ae_output_size=64,
                 in_channels=1,
                 state_size=4,
                 control_size=0):
        super(MPNet, self).__init__()
        self.state_size = state_size
        self.encoder = VoxelEncoder(input_size=ae_input_size, 
                                    output_size=ae_output_size,
                                    in_channels=in_channels)
        self.pnet = PNet(input_size=ae_output_size+state_size * 2,
                         output_size=state_size+control_size)

    def forward(self, x, obs):
        if obs is not None:
            z = self.encoder(obs)
            z_x = torch.cat((z,x), 1)
        else:
            z_x = x
        return self.pnet(z_x)
    
