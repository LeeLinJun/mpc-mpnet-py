import torch
from torch import nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.res1 = ResMLP(input_size, 512)
        self.res2 = ResMLP(512, 256)
        self.res3 = ResMLP(256, 128)
        self.res4 = ResMLP(128, 32)
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.dropout(self.res1(x))
        x = F.dropout(self.res2(x))
        x = F.dropout(self.res3(x))
        x = F.dropout(self.res4(x))
        return self.fc(x)


class ResMLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=False):
        super(ResMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, output_size)
        self.prelu1 = nn.PReLU()

        self.fc11 = nn.Linear(output_size, output_size)
        self.prelu11 = nn.PReLU()

        self.fc12 = nn.Linear(output_size, output_size)
        self.prelu12 = nn.PReLU()

        self.dropout = dropout

    def forward(self, x):
        h = self.prelu1(self.fc1(x))
        if self.dropout:
            h = F.dropout(h)
        dh = self.prelu11(self.fc11(h))
        if self.dropout:
            dh = F.dropout(dh)
        dh = self.prelu12(self.fc12(dh))
        return h + dh
