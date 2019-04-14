import argparse
import itertools
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.optim as optim
import generator
import discriminator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("GPU is available!")

netG = generator.Generator(args.input_nc, args.output_nc)
netD = discriminator.Discriminator(args.input_nc)

if device.type == 'cuda':
    netG.cuda()
    netD.cuda()

ganLoss = nn.MSELoss().to(device)
reconstructionLoss = nn.L1Loss().to(device)
identityLoss = nn.L1Loss().to(device)

G_optimizer = optim.Adam(netG.parameters())
D_optimizer = optim.Adam(netD.parameters())

target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

print("All set, Training Begins!")

for epoch in n_epochs:
    G_loss = []
    D_loss = []
    recLoss1 = []
    recLoss2 = []

    ############################### LOAD DATA BELOW ##################
    # conditioned text at the end of every batch
    # move data to cuda
    for (realA, labelA), (realB, labelB) in zip(trainloaderA, trainloaderB):



    ###############################
    G_optimizer.zero_grad()

    ## from a domain to same domain
    ## source condtion and target condition should be the same

    sameA = netG(realA)
    identityA = identityLoss(sameA, realA)

    sameB = netG(realB)
    identityB = identityLoss(sameB, realB)

    ## GAN Loss
    ## condition the img from source A to target B
    fakeB = netG(realA)
    pred_fakeB = netD(fakeB)
    loss_A2B = ganLoss(pred_fakeB, target_real)

    fakeA = netG(realB)
    pred_fakeA = netD(fakeA)
    loss_B2A = ganLoss(pred_fakeA, target_real)

    ## Reconstruction loss
    recoveredA = netG(fakeB)
    cycle_lossA = reconstructionLoss(realA, recoveredA)

    recoveredB = netG(fakeA)
    cycle_lossB = reconstructionLoss(realB, recoveredB)

    ## Total loss
    G_loss = loss_A2B + loss_B2A + cycle_lossA + cycle_lossB

    G_loss.backward()
    G_optimizer.step()


    ######################################
    D_optimizer.zero_grad()

    ## from D-A
    pred_real1 = netD(realA)
    lossD_real1 =

    pred_fake1 = netD(fakeA.detach())
    lossD_fake1 =

    pred_real2 = netD(realB)
    lossD_real1 =

    pred_fake2 = netD(fakeB.detach())
    lossD_fake2 =


    loss_D_1 = lossD_real1 + lossD_fake1 + lossD_real2 + lossD_fake2

    loss_D_1.backward()
    D_optimizer.step()
