# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main network architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models


def sample_latent(mu, logvar):
    # Reparameterization trick
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(mu)
    return mu + std * eps

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 3, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 3, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 512, 3, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 512, 3, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = spectral_norm(nn.Linear(8192, 256))
        self.logvar = spectral_norm(nn.Linear(8192, 256))
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, 8192)
        return self.mu(h), self.logvar(h)

class SPADE(nn.Module):
    def __init__(self, feature_size, style_size):
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(feature_size, affine=False)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(style_size, 128, 3, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv_gamma = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
    
    def forward(self, x, s):
        s = F.interpolate(s, size=(x.size(2), x.size(3)), mode='nearest')
        s = self.conv(s)
        return self.norm(x) * self.conv_gamma(s) + self.conv_beta(s)

class SPADEResBlk(nn.Module):
    def __init__(self, input_size, output_size, style_size):
        super(SPADEResBlk, self).__init__()
        # Main layer
        self.spade_1 = SPADE(input_size, style_size)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = spectral_norm(nn.Conv2d(input_size, output_size, 3, 1, 1))
        self.spade_2 = SPADE(output_size, style_size)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = spectral_norm(nn.Conv2d(output_size, output_size, 3, 1, 1))
        # Shortcut layer
        self.spade_s = SPADE(input_size, style_size)
        self.relu_s = nn.ReLU(inplace=True)
        self.conv_s = spectral_norm(nn.Conv2d(input_size, output_size, 3, 1, 1))
    
    def forward(self, x, s):
        y  = self.conv_1(self.relu_1(self.spade_1(x, s)))
        y  = self.conv_2(self.relu_2(self.spade_2(y, s)))
        y_ = self.conv_s(self.relu_s(self.spade_s(x, s)))
        return y + y_

class Generator(nn.Module):
    def __init__(self, style_size):
        super(Generator, self).__init__()
        self.fc = spectral_norm(nn.Linear(256, 16384))
        self.spade_resblks = nn.ModuleList([
            # SPADEResBlk(1024, 1024, style_size),  # Seems that the paper describes 1 more unnecessary layer
            SPADEResBlk(1024, 1024, style_size),
            SPADEResBlk(1024, 1024, style_size),
            SPADEResBlk(1024, 512, style_size),
            SPADEResBlk(512, 256, style_size),
            SPADEResBlk(256, 128, style_size),
            SPADEResBlk(128, 64, style_size)
        ])
        self.upsample2x = lambda x: F.interpolate(x, scale_factor=2.0, mode='nearest')
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 3, 3, 1, 1)),
            nn.Tanh(),
        )
    
    def forward(self, x, s):
        h = self.fc(x)
        h = h.view(-1, 1024, 4, 4)
        for spade_resblk in self.spade_resblks:
            h = spade_resblk(h, s)
            h = self.upsample2x(h)
        y = self.conv(h)
        return y

class Discriminator(nn.Module):
    def __init__(self, style_size):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(style_size+3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(512, 1, 4, 1, 0)),
            )
        ])
    
    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        y = [x]
        for layer in self.layers:
            y.append(layer(y[-1]))
        return y[1:]

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    
    def forward(self, x, normalize_input=True):
        assert x.dim() == 4 and x.size(1) == 3, 'Wrong input size {}. Should be (N, 3, H, W)'.format(tuple(x.size()))
        if normalize_input:
            # Normalize inputs
            # from (-1., 1.), i.e., ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # to ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            x = x + 1 / 2
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)
            x =  x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out