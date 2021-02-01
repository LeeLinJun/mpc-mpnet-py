import torch
from torch import nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self, input_size, output_size, dropout=False):
        super(PNet, self).__init__()
        self.res1 = ResMLP(input_size, 512, dropout=dropout)
        self.res2 = ResMLP(512, 256, dropout=dropout)
        self.res3 = ResMLP(256, 128, dropout=dropout)
        self.res4 = ResMLP(128, 32, dropout=dropout)
        self.fc = nn.Linear(32, output_size)

        self.dropout1, self.dropout2, self.dropout3, self.dropout4 = nn.Dropout(), nn.Dropout(), nn.Dropout(), nn.Dropout()

    def forward(self, x):
        x = self.dropout1(self.res1(x))
        x = self.dropout2(self.res2(x))
        x = self.dropout3(self.res3(x))
        x = self.dropout4(self.res4(x))
        return self.fc(x)


class ResMLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=False):
        super(ResMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.LeakyReLU()

        self.fc11 = nn.Linear(output_size, output_size)
        self.relu11 = nn.LeakyReLU()
        self.dropout11 = nn.Dropout()

        self.fc12 = nn.Linear(output_size, output_size)
        self.relu12 = nn.LeakyReLU()
        self.dropout12 = nn.Dropout()


        self.dropout = dropout

    def forward(self, x):
        h = self.relu1(self.fc1(x))
        dh = self.dropout11(self.relu11(self.fc11(h)))
        dh = self.dropout12(self.relu12(self.fc12(dh)))
        return h + dh
