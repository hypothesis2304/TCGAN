import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
# from visdom import Visdom
import numpy as np

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

def hash(domain, inp, bs):
    dim1 = inp.size(2)
    dim2 = inp.size(3)
    if domain == 'A':
        topRows = torch.ones((1, dim1//2, dim2))
        botRows = torch.zeros((1, dim1//2, dim2))
        hashCode = torch.cat((topRows, botRows), 1)
    elif domain == 'B':
        topRows = torch.zeros((1, dim1//2, dim2))
        botRows = torch.ones((1, dim1//2, dim2))
        hashCode = torch.cat((topRows, botRows), 1)
    hashCode = hashCode.unsqueeze(0).repeat(bs,1,1,1)
    return hashCode
