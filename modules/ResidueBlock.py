import torch
import torch.nn as nn

import torch.nn.functional as F

from .WeightNormalizedAddition import *
from .WeightNormalizedConv import *
from .TPReLU import *
from .NearestNeighbourScale2x import *
from .Affine import *

class ResidueBlock(nn.Module):

	def __init__(self, in_channels, out_channels, stride, pad_h, pad_w):
		super(ResidueBlock, self).__init__()

		self.residue = nn.Sequential()
		self.residue.add_module('conv_1', WeightNormalizedConv2d(in_channels, out_channels, stride + 2, stride, (1 + pad_h, 1 + pad_w)))
		self.residue.add_module('tprelu', TPReLU(out_channels))
		self.residue.add_module('conv_2', WeightNormalizedConv2d(out_channels, out_channels, 3, 1, 1))

		self.shortcut = nn.Sequential()
		if pad_h > 0 or pad_w > 0:
			self.shortcut.add_module('pad', nn.ZeroPad2d((pad_w, pad_w, pad_h, pad_h)))
		if stride == 2:
			self.shortcut.add_module('pool', nn.AvgPool2d(2, 2))
		if in_channels != out_channels:
			self.shortcut.add_module('conv', WeightNormalizedConv2d(in_channels, out_channels, 1, 1, 0))

		self.add = WeightNormalizedAddition(2, out_channels, torch.Tensor([0, 1]))
		
		self.post = TPReLU(out_channels)

	def forward(self, input):
		return self.post(self.add(self.residue(input), self.shortcut(input)))

class ResidueBlockTranspose(nn.Module):

	def __init__(self, in_channels, out_channels, stride, pad_h, pad_w, gen_last_block = False):
		super(ResidueBlockTranspose, self).__init__()

		self.residue = nn.Sequential()
		self.residue.add_module('conv_1', WeightNormalizedConvTranspose2d(in_channels, in_channels, 3, 1, 1))
		self.residue.add_module('tprelu', TPReLU(in_channels))
		self.residue.add_module('conv_2', WeightNormalizedConvTranspose2d(in_channels, out_channels, stride + 2, stride, (1 + pad_h, 1 + pad_w)))

		self.shortcut = nn.Sequential()
		if in_channels != out_channels:
			self.shortcut.add_module('conv', WeightNormalizedConvTranspose2d(in_channels, out_channels, 1, 1, 0))
		if stride == 2:
			self.shortcut.add_module('scale', NearestNeighbourScale2x())

		self.add = WeightNormalizedAddition(2, out_channels, torch.Tensor([0, 1]))
		
		if gen_last_block:
			self.post = Affine(out_channels)
		else:
			self.post = TPReLU(out_channels)

		self.pad_h = pad_h
		self.pad_w = pad_w

	def forward(self, input):
		residue = self.residue(input)
		shortcut = self.shortcut(input)
		shortcut = shortcut[:, :, self.pad_h : shortcut.size(2) - self.pad_h, self.pad_w : shortcut.size(3) - self.pad_w]
		return self.post(self.add(residue, shortcut))
