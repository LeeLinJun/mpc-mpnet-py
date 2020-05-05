import torch
from torch import nn

class PNet(nn.Module): #didn't work
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 896), nn.PReLU(), nn.Dropout(),
		nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32), nn.PReLU(),
		nn.Linear(32, output_size))
	def forward(self, x):
		out = self.fc(x)
		return out
