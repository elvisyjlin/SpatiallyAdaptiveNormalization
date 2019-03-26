# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Generate images from trained models"""

import argparse
import json
import os
from os import listdir
from os.path import join
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils as vutils

from networks import Encoder, Generator, sample_latent
from utils import onehot2d

# The synchronized batch normalization is from
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
# from sync_batchnorm import convert_model


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--dataset', type=str, choices=['COCO-Stuff'], default=argparse.SUPPRESS)
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--test_epoch', type=int, default=None)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    # parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    # Arguments
    args = parse()
    print(args)
    
    # Load training setting
    with open(join('results', args.experiment_name, 'setting.json'), 'r', encoding='utf-8') as f:
        setting = json.load(f)
    for key, value in vars(args).items():
        setting[key] = value
    args = argparse.Namespace(**setting)
    print(args)
    
    # Device
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    
    # Paths
    checkpoint_path = join('results', args.experiment_name, 'checkpoint')
    test_path = join('results', args.experiment_name, 'sample_val')
    os.makedirs(test_path, exist_ok=True)
    
    # Data
    if args.dataset == 'COCO-Stuff':
        from data import COCO_Stuff
        val_dset = COCO_Stuff(args.data, mode='val')
        n_classes = COCO_Stuff.n_classes
    val_data = data.DataLoader(val_dset, batch_size = args.batch_size, shuffle=False, drop_last=False)
    
    # Models
    E = Encoder()
    E.to(device)
    G = Generator(n_classes)
    G.to(device)
    
    if args.multi_gpu:  # If trained with multi-GPU, the model needs to be loaded with multi-GPU, too.
        E = nn.DataParallel(E)
        G = nn.DataParallel(G)
        # G = convert_model(G)
    
    # Load from checkpoints
    load_epoch = args.test_epoch
    if load_epoch is None:  # Use the lastest model
        load_epoch = max(int(path.split('.')[0]) for path in listdir(checkpoint_path) if path.split('.')[0].isdigit())
    print('Loading generator from epoch {:03d}'.format(load_epoch))
    E.load_state_dict(torch.load(
        join(checkpoint_path, '{:03d}.E.pth'.format(load_epoch)),
        map_location=lambda storage, loc: storage
    ))
    G.load_state_dict(torch.load(
        join(checkpoint_path, '{:03d}.G.pth'.format(load_epoch)),
        map_location=lambda storage, loc: storage
    ))
    
    E.eval()
    G.eval()
    with torch.no_grad():
        for batch_idx, (reals, annos) in enumerate(tqdm(val_data)):
            reals, annos = reals.to(device), annos.to(device)
            annos_onehot = onehot2d(annos, n_classes).type_as(reals)
            
            # Encode images and sample latents
            mu, logvar = E(reals)
            latents = sample_latent(mu, logvar)
            
            # Generate images
            fakes = G(latents, annos_onehot)
            
            # Save images separately
            for idx in range(reals.size(0)):
                anno = annos[idx].float() / n_classes * 2 - 1
                anno = torch.cat((anno, anno, anno))
                image_out = torch.stack((reals[idx], anno, fakes[idx]))
                vutils.save_image(image_out, join(test_path, '{:04d}.jpg'.format(batch_idx*args.batch_size+idx)), nrow=3, padding=0, normalize=True, range=(-1., 1.))