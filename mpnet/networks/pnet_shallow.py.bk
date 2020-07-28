import torch
from torch import nn
from torch.nn import functional as F

class PNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(PNet, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 32), nn.PReLU(),
		nn.Linear(32, 128), nn.PReLU(), #nn.Dropout(),
		nn.Linear(128, 256), nn.PReLU(),# nn.Dropout(),
		nn.Linear(256, 512), nn.PReLU(), nn.Dropout(),
		nn.Linear(512, 32), nn.PReLU(), nn.Dropout(),
		nn.Linear(32, output_size))

	def forward(self, x):
		out = self.fc(x)
		return out

# class PNet(nn.Module):
# 	def __init__(self, input_size, output_size):
# 		super(PNet, self).__init__()
# 		self.fc1 = nn.Linear(input_size, 32)
# 		self.fc2 = nn.Linear(32, 32)
# 		self.fc3 = nn.Linear(32, 64)
# 		self.fc4 = nn.Linear(64, 64)
# 		self.fc5 = nn.Linear(64, 128)
# 		self.fc6 = nn.Linear(128, 128)
# 		self.fc7 = nn.Linear(128, 128)
# 		self.fc8 = nn.Linear(128, output_size)

# 	def forward(self, x):
# 		# x0 = x
# 		# x1 = F.relu(self.fc1(x0)) # 32
# 		# x2 = F.dropout(F.relu(self.fc2(x1))) # 128
# 		# x3 = F.dropout(F.relu(self.fc3(x2))) # 256
# 		# x4 = F.dropout(F.relu(self.fc4(x3))) # 512

# 		# x5 = F.dropout(F.relu(self.fc5(x4))) + x3 # 256+256
# 		# x6 = F.dropout(F.relu(self.fc6(x5))) + x2 # 128+128
# 		# x7 = F.dropout(F.relu(self.fc7(x6))) + x1 # 32+32
# 		# x8 = F.relu(self.fc8(x7))
# 		x1 = F.relu(self.fc1(x)) # 32
# 		x2 = F.dropout(F.relu(self.fc2(x1))) +x1 # 32+32
# 		x3 = (F.relu(self.fc3(x2)))  # 64
# 		x4 = F.dropout(F.relu(self.fc4(x3))) +x3 # 64 + 64
# 		x5 = (F.relu(self.fc5(x4))) # 128
# 		x6 = F.dropout(F.relu(self.fc6(x5))) + x5 # 128+128
# 		x7 = (F.relu(self.fc7(x6))) + x6 # 128+128
# 		x8 = F.relu(self.fc8(x7))

# 		return x8


# class res_block(nn.Module):
#     def __init__(self, ni, no):
# 		super(res_block, self).__init__()
# 		self.fc1 = nn.Linear(ni, ni)
# 		self.fc2 = nn.Linear(ni, ni)
# 		self.fc3 = nn.Linear(ni, no)
    
# 	def forward(self,x):
#         residual = x
#         out = F.prelu(self.fc1(x))
#         out = F.dropout(F.prelu(self.fc2(out)))
#         out += residual
# 		return self.fc3(out)