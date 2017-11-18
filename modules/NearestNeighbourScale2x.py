import torch
import torch.nn as nn

class NearestNeighbourScale2x(nn.Module):

	def __init__(self):
		super(NearestNeighbourScale2x, self).__init__()

	def forward(self, input):
		return input.unsqueeze(3).unsqueeze(5).expand(input.size(0), input.size(1), input.size(2), 2, input.size(3), 2).contiguous().view(input.size(0), input.size(1), input.size(2) * 2, input.size(3) * 2)