import torch
import torch.nn as nn
import math

from torch.nn import Parameter
import torch.nn.functional as F

class WeightNormalizedLinear(nn.Module):

    def __init__(self, in_features, out_features, scale=False, bias=False, init_factor=1, init_scale=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.Tensor(out_features).fill_(init_scale))
        else:
            self.register_parameter('scale', None)
        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).sqrt().add(1e-8)

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().unsqueeze(0))
        if self.scale is not None:
            output = output.mul(self.scale.unsqueeze(0))
        if self.bias is not None:
            output = output.add(self.bias.unsqueeze(0))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'