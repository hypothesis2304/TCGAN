import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

## The path of the folder with images is not correctly defined
data_dir = "~/data/"

def domainA_laod(domain_type, bs, data_dir):
    ## check the normalizations before training as we are not using any imagenet pretrained weights
    if domain_type == 'train':
        transform = transforms.Compose([transforms.Resize(256),
                                   transforms.RandomHorizontalFlip(),
                				   transforms.RandomRotation(15),
                				   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    elif domain_type == 'test':
        transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        num_workers=0,
        shuffle=True
    )
    return train_loader

import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
