import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

class _WeightNormalizedConvNd(_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride,
			padding, dilation, transposed, output_padding, scale, bias, init_factor, init_scale):
		super(_WeightNormalizedConvNd, self).__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, transposed, output_padding, 1, False)
		if scale:
			self.scale = Parameter(torch.Tensor(self.out_channels).fill_(init_scale))
		else:
			self.register_parameter('scale', None)
		if bias:
			self.bias = Parameter(torch.zeros(self.out_channels))
		else:
			self.register_parameter('bias', None)
		self.weight.data.mul_(init_factor)
		self.weight_norm_factor = 1.0
		if transposed:
			for t in stride:
				self.weight_norm_factor /= t

	def normalized_weight(self):
		weight_norm = self.weight.pow(2)
		for i in range(len(self.kernel_size)):
			weight_norm = weight_norm.sum(i + 2, keepdim = True)
		if self.transposed:
			weight_norm = weight_norm.sum(0, keepdim = True) * self.weight_norm_factor
		else:
			weight_norm = weight_norm.sum(1, keepdim = True)
		weight_norm = weight_norm.sqrt().add(1e-8)
		weight = self.weight.div(weight_norm)
		if self.scale is not None:
			scale_unsqueeze = self.scale
			if self.transposed:
				scale_unsqueeze = scale_unsqueeze.unsqueeze(0)
			else:
				scale_unsqueeze = scale_unsqueeze.unsqueeze(1)
			for i in range(2, weight.dim()):
				scale_unsqueeze = scale_unsqueeze.unsqueeze(i)
			weight = weight.mul(scale_unsqueeze)
		return weight
			
	def __repr__(self):
		s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
			 ', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.scale is None:
			s += ', scale=False'
		if self.bias is None:
			s += ', bias=False'
		s += ')'
		return s.format(name=self.__class__.__name__, **self.__dict__)

class WeightNormalizedConv2d(_WeightNormalizedConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, scale=False, bias=False, init_factor=1, init_scale=1):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(WeightNormalizedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), scale, bias, init_factor, init_scale)

	def forward(self, input):
		return F.conv2d(input, self.normalized_weight(), self.bias, self.stride,
						self.padding, self.dilation, 1)

class WeightNormalizedConvTranspose2d(_ConvTransposeMixin, _WeightNormalizedConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, output_padding=0, scale=False, bias=False, dilation=1, init_factor=1, init_scale=1):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		output_padding = _pair(output_padding)
		super(WeightNormalizedConvTranspose2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			True, output_padding, scale, bias, init_factor, init_scale)

	def forward(self, input, output_size=None):
		output_padding = self._output_padding(input, output_size)
		return F.conv_transpose2d(input, self.normalized_weight(), self.bias, self.stride, self.padding,
			output_padding, 1, self.dilation)
