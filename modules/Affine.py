import torch
import torch.nn as nn

class Affine(nn.Module):

	def __init__(self, nfeature):
		super(Affine, self).__init__()
		self.nfeature = nfeature
		self.scale = nn.Parameter(torch.ones(nfeature))
		self.bias = nn.Parameter(torch.zeros(nfeature))

	def forward(self, input):
		scale = self.scale.unsqueeze(0)
		bias = self.bias.unsqueeze(0)
		while scale.dim() < input.dim():
			scale = scale.unsqueeze(2)
			bias = bias.unsqueeze(2)

		return torch.mul(input, scale.expand_as(input)) + bias.expand_as(input)
		