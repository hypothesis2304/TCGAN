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
