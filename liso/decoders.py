# import torch
# from torch import nn

import mindspore
from mindspore import nn


class BasicDecoder(nn.Cell):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode='pad',
            padding=1
        )

    def _build_models(self):
        self.layers = nn.SequentialCell(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.data_depth)
        )

        return [self.layers]

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self._models = self._build_models()

    def get_feature(self, x):
        res = []
        for model in self._models:
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, mindspore.nn.BatchNorm2d):
                    res.append(x)
        res.append(x)
        return mindspore.ops.cat(res, axis=1)

    def construct(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(mindspore.ops.cat(x_list, axis=1))
                x_list.append(x)

        return x


class DenseDecoder(BasicDecoder):
    """
    The DenseDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    def _build_models(self):
        self.conv1 = nn.SequentialCell(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv2 = nn.SequentialCell(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv3 = nn.SequentialCell(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv4 = nn.SequentialCell(self._conv2d(self.hidden_size * 3, self.data_depth))

        return self.conv1, self.conv2, self.conv3, self.conv4
