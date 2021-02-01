from .voxel_encoder import VoxelEncoder
# from .pnet_shallow import PNet
from .pnet_res import PNet

import torch
from torch import nn


class CostNet(nn.Module):
    def __init__(self, ae_input_size=32, ae_output_size=64,
                 in_channels=1,
                 state_size=4,
                 encoder=None,
                 control_size=0):
        super(CostNet, self).__init__()
        if encoder is None:
            self.encoder = VoxelEncoder(input_size=ae_input_size,
                                        output_size=ae_output_size,
                                        in_channels=in_channels)
        else:
            self.encoder = encoder
        self.mlp = PNet(input_size=ae_output_size+state_size * 2,
                        output_size=1)
        self.aug_ratio = 0.3

    def forward(self, x, obs):
        if obs is not None:
            z = self.encoder(obs)
            z_x = torch.cat((z, x), 1)
        else:
            z_x = x
        return self.mlp(z_x)

    def aug(self, data, label, aug_gt=10, noise=[1, 1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-2]):
        num_aug_sample = int(data.size(0) * self.aug_ratio)
        fake_data = data[:num_aug_sample, :].clone()
        fake_data[:num_aug_sample, 1:] = torch.empty(num_aug_sample, data.size(1) - 1).uniform_(-1, 1) * torch.tensor(noise)
        fake_label = torch.ones(num_aug_sample, 1) * aug_gt

        # print(data.size(), fake_data.size())
        data = torch.cat([data, fake_data], dim=0)
        label = torch.cat([label, fake_label], dim=0)
        return data, label
