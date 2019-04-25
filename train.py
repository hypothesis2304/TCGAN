import argparse
import itertools
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
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

# netG = generator.Generator(args.input_nc, args.output_nc)
netG = generator.Generator(3, num_filter = 32, output_dim=args.output_nc, num_resnet=6)
netD = discriminator.Discriminator(args.input_nc)

if device.type == 'cuda':
    netG.cuda()
    netD.cuda()

ganLoss = nn.MSELoss().to(device)
reconstructionLoss = nn.L1Loss().to(device)
identityLoss = nn.L1Loss().to(device)

G_optimizer = optim.Adam(netG.parameters())
D_optimizer = optim.Adam(netD.parameters())

target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False).to(device)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False).to(device)

print("All set, Training Begins!")

transforms_ = [transforms.Resize(256, Image.BILINEAR),
               transforms.CenterCrop(224),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

dataloader = DataLoader(ImageDataset(data_path, transforms_ = transforms_, unaligned=True), batch_size = batch_size, shuffle=True, num_workers=0, drop_last=True)

print("length: ", len(dataloader)//batch_size)

for epoch in trange(n_epochs, leave=False):
    G_loss = []
    D_loss = []
    recLoss1 = []
    recLoss2 = []

    ############################### LOAD DATA BELOW ##################
    # conditioned text at the end of every batch
    # move data to cuda
    for i, batch in enumerate(dataloader):
        # print(i)
        realA = batch['A']
        realB = batch['B']

        hashAB = utils.hash('A', 'B', realB, batch_size)
        hashBA = utils.hash('B', 'A', realA, batch_size)

        hashAA = utils.hash('A', 'A', realA, batch_size)
        hashBB = utils.hash('B', 'B', realB, batch_size)

        conditionedAB = torch.cat((realA, hashAB), 1)
        conditionedBA = torch.cat((realB, hashBA), 1)

        conditionedAA = torch.cat((realA, hashAA), 1).to(device)
        conditionedBB = torch.cat((realB, hashBB), 1).to(device)

        realA = realA.to(device)
        realB = realB.to(device)

        hashAB = hashAB.to(device)
        hashBA = hashBA.to(device)

        conditionedAB = conditionedAB.to(device)
        conditionedBA = conditionedBA.to(device)

        ###############################
        G_optimizer.zero_grad()

        ## from a domain to same domain
        ## source condtion and target condition should be the same

        sameA = netG(conditionedAA)
        identityA = identityLoss(sameA, realA) * 5.0

        sameB = netG(conditionedBB)
        identityB = identityLoss(sameB, realB) * 5.0

        ## GAN Loss
        ## condition the img from source A to target B
        fakeB = netG(conditionedAB)
        pred_fakeB = netD(fakeB)
        loss_A2B = ganLoss(pred_fakeB, target_real)

        fakeA = netG(conditionedBA)
        pred_fakeA = netD(fakeA)
        loss_B2A = ganLoss(pred_fakeA, target_real)

        ## Reconstruction loss

        fakeConditionedAB = torch.cat((fakeA, hashAB), 1).to(device)
        fakeConditionedBA = torch.cat((fakeB, hashBA), 1).to(device)

        recoveredA = netG(fakeConditionedBA)
        cycle_lossA = reconstructionLoss(realA, recoveredA) * 10.0

        recoveredB = netG(fakeConditionedAB)
        cycle_lossB = reconstructionLoss(realB, recoveredB) * 10.0

        ## Total loss
        G_loss = loss_A2B + loss_B2A + cycle_lossA + cycle_lossB

        G_loss.backward()
        G_optimizer.step()


        ######################################
        D_optimizer.zero_grad()

        ## from D-A
        pred_real1 = netD(realA)
        lossD_real1 = ganLoss(pred_real1, target_real)

        pred_fake1 = netD(fakeA.detach())
        lossD_fake1 = ganLoss(pred_fake1, target_fake)

        pred_real2 = netD(realB)
        lossD_real2 = ganLoss(pred_real2, target_real)

        pred_fake2 = netD(fakeB.detach())
        lossD_fake2 = ganLoss(pred_fake2, target_fake)


        loss_D_1 = lossD_real1 + lossD_fake1 + lossD_real2 + lossD_fake2

        loss_D_1.backward()
        D_optimizer.step()

print("Finished Training!")
torch.save(netG.state_dict(), model_dir + 'common_generator.pkl')
torch.save(netD.state_dict(), model_dir + 'common_discriminator.pkl')
