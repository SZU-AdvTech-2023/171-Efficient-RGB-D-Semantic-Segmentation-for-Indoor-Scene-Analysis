# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


class ConvBN(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(ConvBN, self).__init__()
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2,
                                          bias=False))
        self.add_module('bn', nn.BatchNorm2d(channels_out))


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExcitationTensorRT(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitationTensorRT, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # TensorRT restricts the maximum kernel size for pooling operations
        # by "MAX_KERNEL_DIMS_PRODUCT" which leads to problems if the input
        # feature maps are of large spatial size
        # -> workaround: use cascaded two-staged pooling
        # see: https://github.com/onnx/onnx-tensorrt/issues/333
        if x.shape[2] > 120 and x.shape[3] > 160:
            weighting = F.adaptive_avg_pool2d(x, 4)
        else:
            weighting = x
        weighting = F.adaptive_avg_pool2d(weighting, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def swish(x):
    return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, inplace=True):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.leaky_relu(x,
            negative_slope=self.negative_slope, inplace=self.inplace)

class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.init = init
        self.weight = \
            nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x):
        return nn.functional.prelu(x, self.weight)

class ELU(nn.Module):
    def __init__(self, alpha=1.0, inplace=True):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.elu(x, alpha=self.alpha, inplace=self.inplace)

class SELU(nn.Module):
    def __init__(self, inplace=True):
        super(SELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.selu(x, inplace=self.inplace)
