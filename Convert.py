import torch
import mindspore
from mindspore import save_checkpoint, Tensor
import mindspore.nn as nn
import numpy as np

# 假设 KeNet 是一个在 MindSpore 中定义的模型类
class KeNet(nn.Cell):
    def __init__(self):
        super(KeNet, self).__init__()
        # 定义模型结构，例如：
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # 添加其他层...

    def construct(self, x):
        x = self.conv1(x)
        # 继续构建网络结构
        return x

def convert_pytorch_to_mindspore(pytorch_path, mindspore_path):
    # 加载 PyTorch 模型参数
    pytorch_state_dict = torch.load(pytorch_path, map_location='cpu')['state_dict']

    # 创建 MindSpore 模型实例
    mindspore_model = KeNet()

    # 创建一个字典来存储 MindSpore 格式的参数
    mindspore_params = []

    for name, param in pytorch_state_dict.items():
        # 将 PyTorch 参数转换为 NumPy 数组，然后转换为 MindSpore Tensor
        ms_param = Tensor(param.cpu().numpy())
        mindspore_params.append({'name': name, 'data': ms_param})

    # 保存为 MindSpore 的 checkpoint 文件
    save_checkpoint(mindspore_params, mindspore_path)

# 进行转换
convert_pytorch_to_mindspore("checkpoints/kenet.pth.tar", "checkpoints/kenet.ckpt")
