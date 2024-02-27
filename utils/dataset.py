import os
import torch
import random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def random_crop(image_size=128, crop_scale=(0.5, 0.8)):
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(image_size, scale=crop_scale),
    )


def center_crop(image_size=128, crop_scale=0.65):
    size = int(image_size / crop_scale)
    return torch.nn.Sequential(
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    )

def process_labels(label_list, equalize=True, pos_rate=None):
    positives = []
    negatives = []
    for label in label_list:
        label = label.rstrip()
        filename, label = label.split()
        if label == '1':
            positives.append((filename, 1))
        elif label == '0':
            negatives.append((filename, 0))
        else:
            raise ValueError()

    num_p = len(positives)
    num_n = len(negatives)
    print('Found {:d} positives and {:d} negatives'.format(num_p, num_n))
    if equalize:
        if pos_rate is None:
            rate = num_n // num_p + 1
        else:
            rate = pos_rate
        positives = positives * rate
        random.shuffle(positives)
        positives = positives[:num_n]
        num_p = len(positives)
        num_n = len(negatives)
        print('Equalized to {:d} positives and {:d} negatives'.format(num_p, num_n))

    labels = positives + negatives
    random.shuffle(labels)
    return labels


class HSIDataset(Dataset):

    def __init__(self, images_dir, label_path, image_size=128, crop_scale=(0.5, 0.8), trainset=False,
                 positive_rep_rate=None):
        super(HSIDataset, self).__init__()
        with open(label_path) as f:
            self.labels = f.readlines()
        self.root_dir = images_dir

        if trainset:
            self.labels = process_labels(self.labels, True, positive_rep_rate)
            self.arg_fcn = random_crop(image_size, crop_scale)
        else:
            self.labels = process_labels(self.labels, False)
            self.arg_fcn = center_crop(image_size, (crop_scale[0] + crop_scale[1]) / 2.)
        self.l = len(self.labels)

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        path, label = self.labels[item]

        data = sio.loadmat(os.path.join(self.root_dir, path))['var'].astype(np.float32)
        data = np.transpose(data, [2, 0, 1])
        data = data / np.max(data)
        data = data * 2. - 1.

        data = torch.tensor(data)
        data = self.arg_fcn(data)
        return data, torch.tensor([label + 0.])


class TestDataset(Dataset):

    def __init__(self, images_dir, label_path, image_size=128, crop_scale=0.65):
        super(TestDataset, self).__init__()
        with open(label_path) as f:
            self.labels = f.readlines()
        self.root_dir = images_dir

        self.arg_fcn = center_crop(image_size, crop_scale)
        self.l = len(self.labels)

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        path = self.labels[item].rstrip()
        data = sio.loadmat(os.path.join(self.root_dir, path))['var'].astype(np.float32)
        data = np.transpose(data, [2, 0, 1])
        data = data / np.max(data)
        data = data * 2. - 1.

        data = torch.tensor(data)
        data = self.arg_fcn(data)
        return data