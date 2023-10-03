import os
import re
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import click
import torch.nn as nn
from tqdm import tqdm
# from training.augment import AugmentPipe
try:
    import pyspng
except ImportError:
    pyspng = None


def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def load_cifar(zip_path, npz_path):
    zip_file = zipfile.ZipFile(zip_path)
    all_names = set(zip_file.namelist())
    
    PIL.Image.init()
    image_names = sorted(fname for fname in all_names if file_ext(fname) in PIL.Image.EXTENSION)

    # load labels
    with zip_file.open('dataset.json', 'r') as f:
        labels = json.load(f)['labels']
    
    labels_dict = dict(labels)

    images = []
    labels = []
    
    # load images
    for name in tqdm(image_names):
        with zip_file.open(name, 'r') as f:
            if pyspng is not None and file_ext(name) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)     # HWC => CHW

        # append images
        images.append(image[np.newaxis, :, :, :])

        # append labels
        label = labels_dict[name]
        labels.append(label)

    images = np.concatenate(images, axis=0)
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    # save images and labels
    np.savez(npz_path, images=images, labels=labels)


def load_cifar10_zip(zip_path):
    zip_file = zipfile.ZipFile(zip_path)
    all_names = set(zip_file.namelist())
    
    PIL.Image.init()
    image_names = sorted(fname for fname in all_names if file_ext(fname) in PIL.Image.EXTENSION)

    # load labels
    with zip_file.open('dataset.json', 'r') as f:
        labels = json.load(f)['labels']
    
    labels_dict = dict(labels)

    images = []
    labels = []
    
    # load images
    for name in tqdm(image_names):
        with zip_file.open(name, 'r') as f:
            if pyspng is not None and file_ext(name) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)     # HWC => CHW

        # append images
        images.append(image[np.newaxis, :, :, :])

        # append labels
        label = labels_dict[name]
        labels.append(label)

    images = np.concatenate(images, axis=0)
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    return images, labels


class OptimEDM(nn.Module):
    """
    This class implements the ideal EDM to approximate the denoised function with acceleration
    """
    def __init__(self, 
                 data_path,
                 label_dim = 0,
                 sigma_min = 0,
                 ref_size = None,
                 sigma_max = float('inf'), 
                 device=torch.device('cuda')):
        super().__init__()
        self.data_path = data_path
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
        # load images and labels
        assert data_path.endswith('.zip') or data_path.endswith('.npz')
        if data_path.endswith('.npz'):
            self.data = np.load(data_path)
            self.images, self.labels = self.data['images'], self.data['labels']
        elif data_path.endswith('.zip'):
            self.images, self.labels = load_cifar10_zip(zip_path=data_path)

        self.img_resolution = self.images.shape[2]
        self.img_channels = self.images.shape[1]
        self.label_dim = label_dim

        N, C, H, W = self.images.shape

        # Transform to FloatTensor        
        self.y = torch.from_numpy(self.images).to(torch.float32).to(self.device) / 127.5 - 1  # y shape: (N2, C, H, W)
        self.y = self.y.reshape(N, C*H*W)  # y shape: (N2, C*H*W)
        
        # Enable batch size sampling
        self.data_scale = self.y.shape[0]
        self.ref_size = ref_size
    
    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        if self.ref_size is not None and self.ref_size < self.data_scale:
            random_index = torch.randperm(self.data_scale)[:self.ref_size]
            y = self.y[random_index]
        else:
            y = self.y
        B, C, H, W = x.shape
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        x = x.reshape(B, C*H*W)
        # compute score function
        distance = - torch.cdist(x, y, p=2) ** 2 / 2 / sigma ** 2     # (B, N)
        prob = torch.softmax(distance, dim=1)                         # softmax in the axis of N, (B, N)
        denoised = prob @ y                                           # (B, C*H*W)
        denoised = denoised.reshape(B, C, H, W)
        # score = (denoised - x) / sigma ** 2                                                    
        return denoised
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
