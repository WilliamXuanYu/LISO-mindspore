import numpy as np
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import ImageReadMode

# import torch
# import torchvision
# from torchvision import transforms

import mindspore
from mindspore import dataset

from mindspore.nn import BCEWithLogitsLoss
from mindspore.dataset.vision import RandomCrop, CenterCrop, Decode
from mindspore import load_checkpoint, load_param_into_net, context

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]


def get_default_transform(crop_size):
    return mindspore.dataset.transforms.Compose([
        mindspore.dataset.vision.RandomHorizontalFlip(),
        mindspore.dataset.vision.RandomCrop(crop_size, pad_if_needed=True),
        mindspore.dataset.vision.ToTensor(),
        mindspore.dataset.vision.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
    ])


EVAL_TRANSFORM = mindspore.dataset.transforms.Compose([
    mindspore.dataset.vision.ToTensor(),
    mindspore.dataset.vision.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])


class ImageFolder(dataset.ImageFolderDataset):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


# class DataLoader(torch.utils.data.DataLoader):
#     def __init__(
#         self,
#         path,
#         limit=np.inf,
#         shuffle=True,
#         batch_size=4,
#         train=True,
#         crop_size=360,
#         num_workers=8,
#         *args, **kwargs):
#
#         transform = get_default_transform(crop_size) if train else EVAL_TRANSFORM
#
#         super().__init__(
#             ImageFolder(path, transform, limit),
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             *args,
#             **kwargs
#         )

import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from PIL import Image
import os


def get_default_transform(crop_size):
    return [
        vision.Resize(crop_size),
        vision.RandomCrop(crop_size),
        vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
        vision.HWC2CHW()
    ]


EVAL_TRANSFORM = [
    vision.Resize(360),
    vision.CenterCrop(360),
    vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
    vision.HWC2CHW()
]


class DataLoader:
    def __init__(
            self,
            path,
            limit=np.inf,
            shuffle=True,
            batch_size=4,
            train=True,
            crop_size=360):

        transform = get_default_transform(crop_size) if train else EVAL_TRANSFORM

        # 使用自定义的 ImageFolder 逻辑加载数据
        data_set = self.create_image_dataset(path, transform, limit)

        self.data_loader = data_set.batch(batch_size, drop_remainder=True)

        if shuffle:
            self.data_loader = self.data_loader.shuffle(buffer_size=batch_size * 10)

    def create_image_dataset(self, path, transform, limit):
        images = []
        labels = []

        # 遍历文件夹加载图像和标签
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(root, file)
                    label = os.path.basename(root)

                    # img = vision.read_image(img_path, ImageReadMode.UNCHANGED)
                    img = np.fromfile(img_path, np.uint8)
                    # print(type(img))
                    # print(type(label))

                    images.append(img)
                    labels.append(label)
                    if len(images) >= limit:
                        break
            if len(images) >= limit:
                break

        # 将图像路径和标签转换为数据集

        data = list(zip(images, labels))
        dataset = ds.GeneratorDataset(data, column_names=["image", "label"], shuffle=False)

        # 应用数据增强和转换
        dataset = dataset.map(operations=vision.Decode(), input_columns=["image"])
        dataset = dataset.map(operations=transform, input_columns=["image"])

        # 打印前几个样本
        # iterator = dataset.create_dict_iterator(output_numpy=True)
        #
        # for i, batch in enumerate(iterator):
        #     print(f"Sample {i}:")
        #     print("Image:", batch['image'])
        #     print("Label:", batch['label'])
        #     if i >= 5:  # 只打印前5个样本，防止输出过多
        #         break

        return dataset

    def __iter__(self):
        return self.data_loader.create_dict_iterator(output_numpy=False)

# 使用示例
# loader = MindSporeDataLoader('path/to/data', batch_size=4, train=True)
# for batch in loader:
#     images, labels = batch["image"], batch["label"]
