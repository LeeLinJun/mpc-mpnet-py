from .voxel_encoder import VoxelEncoder
from .pnet import PNet

import torch
from torch import nn

class VAEMPNet(nn.Module):
    def __init__(self, ae_input_size=32, ae_output_size=64, code_size=32,
                 in_channels=1,
                 state_size=4,
                 control_size=0):
        super(VAEMPNet, self).__init__()
        self.encoder = VoxelEncoder(input_size=ae_input_size, 
                                    output_size=ae_output_size,
                                    in_channels=in_channels)
        self.pnet = PNet(input_size=ae_output_size+state_size * 2,
                         output_size=state_size+control_size)
        
        self.vae_encoder_mean = nn.Linear(ae_output_size+state_size * 2, code_size)
        self.vae_encoder_var = nn.Linear(ae_output_size+state_size * 2, code_size)

        self.vae_decoder = nn.Linear(code_size, ae_output_size+state_size * 2)
        
    def forward(self, x, obs, code_var=1):
        if obs is not None:
            z = self.encoder(obs)
            z_x = torch.cat((z,x), 1)
        else:
            z_x = x
            
        mu = self.vae_encoder_mean(z_x)
        logvar = self.vae_encoder_var(z_x)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * code_var
        z_x_encoded = mu + eps*std
        z_x = self.vae_decoder(z_x_encoded)
        return self.pnet(z_x)
