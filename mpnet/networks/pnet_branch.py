import torch
from torch import nn

class PNet(nn.Module):
	def __init__(self, input_size, output_size=(2, 2)):
		assert len(output_size) == 2
		super(PNet, self).__init__()
		self.snet = nn.Sequential(
		nn.Linear(input_size, 32), nn.PReLU(),
		nn.Linear(32, 128), nn.PReLU(), #nn.Dropout(),
		nn.Linear(128, 256), nn.PReLU(),# nn.Dropout(),
		nn.Linear(256, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 32), nn.PReLU(), nn.Dropout(),
		nn.Linear(32, output_size[0]))

		self.vnet = nn.Sequential(
		nn.Linear(input_size, 32), nn.PReLU(),
		nn.Linear(32, 128), nn.PReLU(), #nn.Dropout(),
		nn.Linear(128, 256), nn.PReLU(),# nn.Dropout(),
		nn.Linear(256, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 32), nn.PReLU(), nn.Dropout(),
		nn.Linear(32, output_size[1]))

	def forward(self, x):
		s = self.snet(x)
		v = self.vnet(x)
		return  torch.cat(
			(
				s[:,0].unsqueeze(1),
				v[:,0].unsqueeze(1),
				s[:,1].unsqueeze(1),
				v[:,1].unsqueeze(1)
			), 1)
