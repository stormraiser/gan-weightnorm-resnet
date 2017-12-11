import torch
import torch.nn as nn
import math

from torch.nn import Parameter

class WeightNormalizedAddition(nn.Module):

	def __init__(self, in_operands, in_channels, initial_weight = None):
		super(WeightNormalizedAddition, self).__init__()
		if initial_weight is None:
			initial_weight = torch.Tensor(in_operands).fill(1 / math.sqrt(in_operands))
		self.weight = Parameter(initial_weight.unsqueeze(1).expand(in_operands, in_channels).contiguous())

	def forward(self, *inputs):
		weight_norm = self.weight.pow(2).sum(0).sqrt().add(1e-8)
		output = None
		for i in range(len(inputs)):
			current_weight = torch.div(self.weight[i], weight_norm).unsqueeze(0)
			for j in range(2, inputs[i].dim()):
				current_weight = current_weight.unsqueeze(j)
			if output is None:
				output = torch.mul(inputs[i], current_weight)
			else:
				output = output + torch.mul(inputs[i], current_weight)
		return output
