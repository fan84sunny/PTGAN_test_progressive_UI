import glob
import math
import numbers
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import transforms


def load_all_landmark(gallery_pose_list):
    landmark_dict = dict()
    for type in gallery_pose_list:
        for gallery_pose in type:
            for file in gallery_pose:
                landmark = []
                with open(file, 'r') as f:
                    landmark_file = f.readlines()
                size = Image.open(file[:-4] + '.jpg').size
                for i, line in enumerate(landmark_file):
                    if i % 2 == 0:
                        h0 = int(float(line) * 224 / size[0])
                        if h0 < 0:
                            h0 = -1
                    else:
                        w0 = int(float(line) * 224 / size[1])
                        if w0 < 0:
                            w0 = -1
                        landmark.append(torch.Tensor([[w0, h0]]))
                landmark = torch.cat(landmark).long()
                # avoid to over fit
                ram = random.randint(0, 19)
                landmark[ram][0] = random.randint(0, 224 - 1)
                landmark[ram][1] = random.randint(0, 224 - 1)
                landmark_dict[file] = landmark
    return landmark_dict


def _pluck(root, query):
    ret = []
    if query:
        path = glob.glob(root + '/0002_c002_00030600_0.jpg')
        index = -1
        for fname in path:
            pid = int(fname[-24:-20])
            camid = int(fname[-18:-15])
            ret.append((fname, pid, camid, -1))
    else:
        path = glob.glob(root + '/*/*/*.jpg')
        frame2trackID = dict()
        with open(root + '/../test_track.txt') as f:
            for track_id, line in enumerate(f.readlines()):
                curLine = line.strip().split(" ")
                for frame in curLine:
                    frame2trackID[frame] = track_id

        for fname in path:
            pid = int(fname[-24:-20])
            camid = int(fname[-18:-15])
            ret.append((fname, pid, camid, frame2trackID[fname[-24:]]))
    return ret


def get_pose_list(root):
    pose_list = [[[] for i in range(8)] for j in range(9)]
    path = glob.glob(root + '/*/*/*.jpg')

    for fname in path:
        type = int(fname[-28])
        pose = int(fname[-26])
        pose_file = fname[:-4] + '.txt'
        pose_list[type][pose].append(pose_file)
    return pose_list


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, height=256, width=256, pose_aug='no'):
        self.height = height
        self.width = width
        self.dataset = dataset
        self.transform = transform
        self.pose_aug = pose_aug

        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RectScale(height, width),
                transforms.ToTensor(),
                normalizer,
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_single_item_with_pose(index)

    def _get_single_item(self, index):
        fname, pid, pose, camid, color, type = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid, color, type

    def _get_single_item_with_pose(self, index):
        fname, pid, camid, trackid = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        return {'origin': img,
                'pid': pid,
                'camid': camid,
                'trackid': trackid,
                'file_name': fname
                }


def get_data(data_dir):
    query_root = osp.join(data_dir, 'query')
    train_root = osp.join(data_dir, 'train')
    gallery_root = osp.join(data_dir, 'test')
    # query_data = _pluck(query_root, True)
    gallery_data = _pluck(gallery_root, False)
    gallery_pose_list = get_pose_list(train_root)
    # query_dataset = ImageDataset(query_data)
    gallery_dataset = ImageDataset(gallery_data)
    # use combined trainval set for training as default
    # query_loader = DataLoader(query_dataset, batch_size=32, num_workers=8, pin_memory=True)
    query_loader = None
    gallery_loader = DataLoader(gallery_dataset, batch_size=8, num_workers=8, pin_memory=True)

    return query_loader, gallery_loader, gallery_pose_list


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, device=0, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, landmark):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        maps = torch.zeros(len(landmark), 20, 224, 224).to(self.device)
        value = torch.min(landmark, dim=2).values
        flag = value != -1
        index = landmark[flag].cpu().detach().numpy().astype(np.int)
        # index = index
        # maps[flag][:, index[:, 0], index[:, 1]] = 1
        # maps[flag][:, index[:, 0], index[:, 1]] = tmp
        tmp = maps[flag]
        for i, (x, y) in enumerate(index):
            tmp[i, x, y] = 1
        maps[flag] = tmp
        maps = self._generate_pose_map_tensor(maps)
        m = torch.max(maps.view(len(landmark), 20, -1), dim=2).values
        maps[flag] = maps[flag] / m[flag][:, None, None]
        # maps = maps / maps.max()
        # maps = np.stack(maps, axis=0)
        # map_list.append(maps)
        return maps

    def _generate_pose_map_tensor(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups, padding=10)
