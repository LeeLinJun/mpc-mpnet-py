from .voxel_encoder_smaller import VoxelEncoder
from .pnet import PNet

import torch
from torch import nn

class MPNet(nn.Module):
    def __init__(self, ae_input_size=32, ae_output_size=64,
                 in_channels=32,
                 state_size=13,
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
        pred = self.pnet(z_x)
        output = pred.clone() #torch.zeros(pred.size()).to(pred.get_device())
        output[:, :3] = pred[:, :3]
        output[:, 3:7] = pred[:, 3:7] / torch.norm(pred[:, 3:7], dim=1, keepdim=True, p=2).clamp(min=1e-4)
        output[:, 7:] = pred[:, 7:]
        return output
    
    # def aug(self, data, label, noise=[1e-2]*26):
    #     data[:, 1:] += torch.empty(data[:, 1:].size(0), data[:, 1:].size(1)).uniform_(-1, 1) * torch.tensor(noise)
    #     return data, label
