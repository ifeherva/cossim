"""
Sharpened Cosine Distance implementation based on keras code published here:
https://www.rpisoni.dev/posts/cossim-convolution/

Written by Istvan Fehervari
"""
import collections
import math
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def _reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))


_pair = _ntuple(2)


class CosSim2D(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=True,
                 depthwise_separable=False,
                 padding=0,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
        super(CosSim2D, self).__init__()

        self.kernel_size = kernel_size
        kernel_size_ = _pair(kernel_size)
        self.stride = stride
        self.padding_mode = padding_mode
        self.depthwise_separable = depthwise_separable

        self.padding = padding if isinstance(padding, str) else _pair(padding)
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * 2
            if padding == 'same':
                for k, i in zip(kernel_size_, range(1, -1, -1)):
                    total_padding = (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.clip = self.kernel_size // 2

        if depthwise_separable:
            self.weight = nn.Parameter(torch.empty(
                (1, kernel_size * kernel_size, out_channels), device=device, dtype=dtype))
        else:
            self.weight = nn.Parameter(torch.empty(
                    (1, in_channels * kernel_size * kernel_size, out_channels), device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.p = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        nn.init.constant_(self.p, 2)

    def sigplus(self, x):
        return x.sigmoid() * F.softplus(x)

    def _forward_once(self, x):
        out_x = math.ceil((x.shape[2] - 2 * self.clip) / self.stride)
        out_y = math.ceil((x.shape[3] - 2 * self.clip) / self.stride)

        x = F.unfold(x, self.kernel_size, stride=self.stride)
        x_norm = torch.linalg.vector_norm(x, dim=1, ord=2, keepdim=True)
        x_norm = torch.maximum(x_norm, torch.ones_like(x_norm) * 1e-6)

        k_norm = torch.linalg.vector_norm(self.weight, dim=1, ord=2, keepdim=True)
        k_norm = torch.maximum(k_norm, torch.ones_like(k_norm) * 1e-6)

        x = torch.matmul((x / x_norm).permute(0, 2, 1), self.weight / k_norm)

        sign = torch.sign(x)
        x = torch.abs(x) + 1e-12
        x = torch.pow(x + self.sigplus(self.bias), self.sigplus(self.p))
        x = sign * x
        x = F.fold(x.permute(0, 2, 1), (out_x, out_y), kernel_size=1)
        return x

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            x = F.pad(x, self._reversed_padding_repeated_twice, mode='constant', value=0)

        if self.depthwise_separable:
            x = torch.stack([self._forward_once(x[:, i, None, ...]) for i in range(x.shape[1])])
            s = x.shape
            x = x.permute(1, 0, 2, 3, 4).reshape(-1, s[0]*s[2], *s[3:])
        else:
            x = self._forward_once(x)

        return x
