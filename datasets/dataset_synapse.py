import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)  # 随机选择旋转次数 (0 到 3 次 90 度旋转)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2) # 随机选择翻转的轴 (水平翻转或垂直翻转)
    image = np.flip(image, axis=axis).copy() # 翻转图像
    label = np.flip(label, axis=axis).copy() # 翻转标签
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20) # 随机选择旋转角度 (-20 到 20 度)
    image = ndimage.rotate(image, angle, order=0, reshape=False) # 旋转图像
    label = ndimage.rotate(label, angle, order=0, reshape=False) # 旋转标签
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size # 输出图像的尺寸

    def __call__(self, sample):
        image, label = sample['image'], sample['label'] # 获取图像和标签

        # 随机选择旋转、翻转或不做处理
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # 如果图像尺寸与目标尺寸不同，进行缩放
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # 缩放图像，使用三阶插值  self.output_size[0] 是目标高度，x 是原始图像的高度。
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 缩放标签，使用最近邻插值
        # 将 numpy 数组转换为 PyTorch 张量，并添加通道维度
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # 增加一个维度 (C, H, W)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample




class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # 数据增强/变换
        self.split = split  # 数据集划分 ('train', 'test', etc.)
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  # 从文本文件中读取样本列表
        self.data_dir = base_dir  # 数据存储的目录

    def __len__(self):
        return len(self.sample_list)  # 返回数据集的样本数量

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')  # 获取样本名
            data_path = os.path.join(self.data_dir, slice_name+'.npz')  # 获取样本的路径
            data = np.load(data_path)  # 加载 .npz 文件
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')  # 测试集时获取体数据名
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)  # 加载 HDF5 文件
            image, label = data['image'][:], data['label'][:]

        # 训练时进行数据增强
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # 将标签转换为 one-hot 编码，并调整维度为 (C, H, W)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


