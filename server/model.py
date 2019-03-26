# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Model that takes one pair of a style image and a semantic image"""

import argparse
import json
import numpy as np
from os import listdir
from os.path import join

import torch
import torch.nn as nn

from ..networks import Encoder, Generator, sample_latent
from ..utils import onehot2d

# The synchronized batch normalization is from
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
# from sync_batchnorm import convert_model


class Model(nn.Module):
    def __init__(self, experiment_name, load_epoch):
        super(Model, self).__init__()
        
        # Load training setting
        with open(join('results', experiment_name, 'setting.json'), 'r', encoding='utf-8') as f:
            setting = json.load(f)
        args = argparse.Namespace(**setting)
        
        # Device
        self.device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
        
        # Paths
        checkpoint_path = join('results', experiment_name, 'checkpoint')
        
        # Models
        if args.dataset == 'COCO-Stuff':
            from ..data import COCO_Stuff
            self.n_classes = COCO_Stuff.n_classes
        self.E = Encoder()
        self.E.to(self.device)
        self.G = Generator(self.n_classes)
        self.G.to(self.device)

        if args.multi_gpu:
            self.E = nn.DataParallel(self.E)
            self.G = nn.DataParallel(self.G)
            # self.G = convert_model(self.G)
    
        # Load from checkpoints
        if load_epoch is None:  # Use the lastest model
            load_epoch = max(int(path.split('.')[0]) for path in listdir(checkpoint_path) if path.split('.')[0].isdigit())
        print('Loading generator from epoch {:03d}'.format(load_epoch))
        self.E.load_state_dict(torch.load(
            join(checkpoint_path, '{:03d}.E.pth'.format(load_epoch)),
            map_location=lambda storage, loc: storage
        ))
        self.G.load_state_dict(torch.load(
            join(checkpoint_path, '{:03d}.G.pth'.format(load_epoch)),
            map_location=lambda storage, loc: storage
        ))
        self.E.eval()
        self.G.eval()
    
    def forward(self, img, ann):
        # Two numpy arrays
        assert img.ndim >= 2 and ann.ndim == 2 and img.shape[0] == ann.shape[0] and img.shape[1] == ann.shape[1]
        # Preprocess image and annotation
        img = torch.from_numpy(np.array(img))
        ann = torch.from_numpy(np.array(ann))
        if img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        else:
            img = img.transpose(0, 1).transpose(0, 2)
        ann = ann.unsqueeze(0)
        img = img.float().div(255).mul(2).add(-1)
        ann = ann.long()
        ann = ann + 1
        ann[ann==(255+1)] = 0
        # Move to GPU
        img, ann = img.to(self.device), ann.to(self.device)
        img = img.unsqueeze(0)
        ann = ann.unsqueeze(0)
        ann = onehot2d(ann, self.n_classes).type_as(img)
        # Forward
        with torch.no_grad():
            mu, logvar = self.E(img)
            latents = sample_latent(mu, logvar)
            out = self.G(latents, ann)
        # Denormalize
        out = out.detach().cpu().squeeze(0)
        out = out.add(1).div(2).mul(255)
        out = torch.clamp(out, min=0, max=255).type(torch.uint8)
        out = out.transpose(0, 2).transpose(0, 1)
        return out.numpy()