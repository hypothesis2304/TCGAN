import torch.nn as nn
import torch.nn.functional as F
import time
import torch

class ImageSizeAdjustBlock(nn.Module):
    def __init__(self,in_channels=5, out_channels=3, kernel_size =3, stride=1, padding=1, activation='relu'):
        super(ImageSizeAdjustBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ReLU(True)
    def forward(self, x):
        out = self.instance_norm(self.conv(x))
        out1 = self.activation(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, activation='relu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding)
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()
        conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        bn = nn.InstanceNorm2d(num_filter)
        relu = nn.ReLU(True)
        pad = nn.ReflectionPad2d(1)

        self.resnet_block = nn.Sequential(
            pad,
            conv1,
            bn,
            relu,
            pad,
            conv2,
            bn
        )

    def forward(self, x):
        out = self.resnet_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, num_resnet):
        super(Generator, self).__init__()

        # Reflection padding
        self.pad = nn.ReflectionPad2d(3)
        # Encoder
        self.conv0 = ImageSizeAdjustBlock()
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(num_resnet):
            self.resnet_blocks.append(ResnetBlock(num_filter * 4))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 4, num_filter * 2)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter)
        self.deconv3 = ConvBlock(num_filter, output_dim,
                                 kernel_size=7, stride=1, padding=0, activation='tanh', batch_norm=False)

    def forward(self, x):
        # Encoder
        pad1 = self.pad(x)
        adjImg = self.conv0(pad1)
        enc1 = self.conv1(adjImg)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        # Resnet blocks
        res = self.resnet_blocks(enc3)
        # Decoder
        dec1 = self.deconv1(res)
        dec2 = self.deconv2(dec1)
        out = self.deconv3(self.pad(dec2))
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                nn.init.normal(m.deconv.weight, mean, std)
            if isinstance(m, ResnetBlock):
                nn.init.normal(m.conv.weight, mean, std)
                nn.init.constant(m.conv.bias, 0)

#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
#
#         conv_block = [  nn.ReflectionPad2d(1),
#                         nn.Conv2d(in_features, in_features, 3),
#                         nn.InstanceNorm2d(in_features),
#                         nn.ReLU(inplace=True),
#                         nn.ReflectionPad2d(1),
#                         nn.Conv2d(in_features, in_features, 3),
#                         nn.InstanceNorm2d(in_features)  ]
#
#         self.conv_block = nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         return x + self.conv_block(x)
#
# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(Generator, self).__init__()
#
#         # Initial convolution block
#         model = [   nn.ReflectionPad2d(3),
#                     nn.Conv2d(input_nc, 64, 7),
#                     nn.InstanceNorm2d(64),
#                     nn.ReLU(inplace=True) ]
#
#         # Downsampling
#         in_features = 64
#         out_features = in_features*2
#         for _ in range(2):
#             model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features*2
#
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]
#
#         # Upsampling
#         out_features = in_features//2
#         for _ in range(2):
#             model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features//2
#
#         model += [  nn.ReflectionPad2d(3),
#                     nn.Conv2d(64, output_nc, 7),
#                     nn.Tanh() ]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         return self.model(x)
