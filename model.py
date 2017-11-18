import torch
import torch.nn as nn
import torch.nn.functional as F

import modules

class Discriminator(nn.Module):

	def __init__(self, w_in, h_in, num_features, num_blocks):
		super(Discriminator, self).__init__()

		f_prev = 3
		w = w_in
		h = h_in

		self.net = nn.Sequential()

		for i in range(len(num_features)):
			f = num_features[i]
			if i == len(num_features) - 1:
				pad_w = 0
				pad_h = 0
			else:
				if (w % 4 == 2):
					pad_w = 1
				else:
					pad_w = 0
				if (h % 4 == 2):
					pad_h = 1
				else:
					pad_h = 0
			for j in range(num_blocks[i]):
				if j == 0:
					self.net.add_module('level_{0}_block_{1}'.format(i, j), modules.ResidueBlock(f_prev, f, 2, pad_h, pad_w))
				else:
					self.net.add_module('level_{0}_block_{1}'.format(i, j), modules.ResidueBlock(f, f, 1, 0, 0))
			f_prev = f
			w = (w + pad_w * 2) // 2
			h = (h + pad_h * 2) // 2

		self.final = modules.WeightNormalizedConv2d(f_prev, 1, (h, w), 1, 0, scale = True, bias = True)

	def forward(self, input):
		return self.final(self.net(input)).contiguous().view(input.size(0))

class Generator(nn.Module):

	def __init__(self, w_out, h_out, num_features, num_blocks, code_size):
		super(Generator, self).__init__()

		pad_w = []
		pad_h = []
		w = w_out
		h = h_out
		for i in range(len(num_features) - 1):
			if (w % 4 == 2):
				pad_w.append(1)
				w = (w + 2) // 2
			else:
				pad_w.append(0)
				w = w // 2
			if (h % 4 == 2):
				pad_h.append(1)
				h = (h + 2) // 2
			else:
				pad_h.append(0)
				h = h // 2
		w = w // 2
		h = h // 2
		pad_w.append(0)
		pad_h.append(0)

		self.net = nn.Sequential()

		self.initial_fc = modules.WeightNormalizedLinear(code_size, num_features[-1] * h * w, scale = True, bias = True, init_factor = 0.01)
		self.initial_size = (num_features[-1], h, w)
		self.initial_prelu = nn.PReLU(num_features[-1])

		for i in range(len(num_features)):
			level = len(num_features) - 1 - i
			f = num_features[level]
			if level == 0:
				f_next = 3
			else:
				f_next = num_features[level - 1]
			for j in range(num_blocks[level]):
				if j == num_blocks[level] - 1:
					self.net.add_module('level_{0}_block_{1}'.format(level, j), modules.ResidueBlockTranspose(f, f_next, 2, pad_h[level], pad_w[level], gen_last_block = (level == 0)))
				else:
					self.net.add_module('level_{0}_block_{1}'.format(level, j), modules.ResidueBlockTranspose(f, f, 1, 0, 0))

	def forward(self, input):
		return F.sigmoid(self.net(self.initial_prelu(self.initial_fc(input).contiguous().view(input.size(0), *self.initial_size))))
