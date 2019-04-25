import itertools
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch
import torch.optim as optim
import generator
import discriminator
import torch.nn as nn
from torch import Tensor
from datasets import ImageDataset
import utils
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument('--dataroot', type=str, default="./datasets/facades/", help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
args = parser.parse_args()
print(args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = "./datasets/facades/"
save_dir = "./datasets/facades/" + '_results/'
model_dir = "./datasets/facades/" + '_model/'

batch_size = args.batchSize
n_epochs = args.n_epochs

if torch.cuda.is_available():
    print("GPU is available!")

netG = generator.Generator(3, num_filter = 32, output_dim=args.output_nc, num_resnet=6)
netD = discriminator.Discriminator(args.input_nc)

if device.type == 'cuda':
    netG.cuda()
    netD.cuda()

netG.load_state_dict(torch.load(model_dir + 'common_generator.pkl'))
netD.load_state_dict(torch.load(model_dir + 'common_discriminator.pkl'))


transforms_ = [transforms.Resize(256),
               transforms.RandomCrop(224),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

dataloader = DataLoader(ImageDataset(data_path, transforms_ = transforms_, unaligned=True), batch_size = batch_size, shuffle=True, num_workers=0, drop_last=True)

print("Testing Begins!!")
with torch.no_grad():
    for i,batch in enumerate(dataloader):
        realA = batch['A'].to(device)
        realB = batch['B'].to(device)
        hashBA = utils.hash('B', 'A', realB, batch_size).to(device)
        hashAB = utils.hash('A', 'B', realA, batch_size).to(device)
        conditionedAB = torch.cat((realA, hashAB), 1).to(device)
        conditionedBA = torch.cat((realB, hashBA), 1).to(device)

        fakeB = netG(conditionedAB)
        fakeConditionedBA = torch.cat((fakeB, hashBA), 1).to(device)
        recon_A = netG(fakeConditionedBA)

        viz1 = realA[0,:,:,:].to('cpu')
        viz = fakeB[0,:,:,:].to('cpu')
        viz2 = recon_A[0,:,:,:].to('cpu')
        break

plt.imshow(transforms.ToPILImage()(viz2), interpolation="bicubic")
# plt.imshow(transforms.ToPILImage()(viz1), interpolation="bicubic")
plt.show()


#     utils.plot_test_results(realA, fakeB, recon_A, i, save=True, save_dir=save_dir + 'AtoB/')
#     print('%d images are generated.' % (i + 1))
#
# print("Testing Done!!")
# print("Result Images stored in the results Dir")
