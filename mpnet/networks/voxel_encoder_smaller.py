import torch
from torch import nn

class VoxelEncoder(nn.Module):
    def __init__(self, input_size, output_size, in_channels):
        super(VoxelEncoder, self).__init__()
        input_size = [input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.output_size = output_size

        with torch.no_grad():
            x = self.encoder(torch.autograd.Variable(torch.rand([1, in_channels] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 64),
            nn.PReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
