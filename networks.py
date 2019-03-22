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


def sample_latent(mu, logvar):
    # Reparameterization trick
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(mu)
    return mu + std * eps

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(8192, 256)
        self.logvar = nn.Linear(8192, 256)
    
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
        self.upsample = lambda x: F.interpolate(x, scale_factor=2.0, mode='nearest')
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 3, 3, 1, 1)),
            nn.Tanh(),
        )
    
    def forward(self, x, s):
        h = self.fc(x)
        h = h.view(-1, 1024, 4, 4)
        for spade_resblk in self.spade_resblks:
            h = spade_resblk(h, s)
            h = self.upsample(h)
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
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.InstanceNorm2d(64),
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