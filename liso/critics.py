# import torch
# from torch import nn

import mindspore
from mindspore import nn

class BasicCritic(nn.Cell):
    """
    The BasicCritic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).

    Input: (N, 3, H, W)
    Output: (N, 1)
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
        )

    # 这里提到的LeakyReLU(),原本使用的是torch.nn.LeakyReLU(inplace=True),但是mindspore没有inplace参数
    def _build_models(self):
        return nn.SequentialCell(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, 1)
        )

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def forward(self, x):
        x = self._models(x)
        x = mindspore.ops.mean(x.view(x.size(0), -1), axis=1)

        return x
