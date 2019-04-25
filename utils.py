import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import random

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

def hash(src, tar, inp, bs):
    dim1 = inp.size(2)
    dim2 = inp.size(3)
    if src == 'A' and tar == 'B':
        topRows = torch.ones((1, dim1//2, dim2))
        botRows = torch.zeros((1, dim1//2, dim2))
        hashCode = torch.cat((topRows, botRows), 1)
    elif src == 'B' and tar == 'A':
        topRows = torch.zeros((1, dim1//2, dim2))
        botRows = torch.ones((1, dim1//2, dim2))
        hashCode = torch.cat((topRows, botRows), 1)
    elif src == 'A' and tar == 'A':
        topRows = torch.ones((1, dim1//2, dim2))
        botRows = torch.ones((1, dim1//2, dim2))
        hashCode = torch.cat((topRows, botRows), 1)
    elif src == 'B' and tar == 'B':
        topRows = torch.zeros((1, dim1//2, dim2))
        botRows = torch.zeros((1, dim1//2, dim2))
        hashCode = torch.cat((topRows, botRows), 1)

    hashCode = hashCode.unsqueeze(0).repeat(bs,1,1,1)
    return hashCode

def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
