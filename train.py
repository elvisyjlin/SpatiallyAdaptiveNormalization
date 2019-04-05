# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main entry point to train a model"""

import argparse
import datetime
import itertools
import json
import os
from os.path import join
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torchsummary import summary

from networks import Encoder, Generator, Discriminator, VGG, sample_latent
from utils import onehot2d

# The synchronized batch normalization is from
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch.git
# from sync_batchnorm import convert_model


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trainable(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)

def init_weights(m):
    if type(m) is nn.Linear:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
    elif type(m) is nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/COCO-Stuff')
    # '/share/data/COCO-Stuff'
    parser.add_argument('--dataset', type=str, choices=['COCO-Stuff'], default='COCO-Stuff')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambda_fm', type=float, default=10.0)
    parser.add_argument('--lambda_kl', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_decay', type=int, default=100)
    parser.add_argument('--lr_G', type=float, default=0.0001)
    parser.add_argument('--lr_D', type=float, default=0.0004)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_iters', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--sample_epochs', type=int, default=1)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--experiment_name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%dM%H:%M.%f"))
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--load_from_experiment', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    # Arguments
    args = parse()
    print(args)
    
    # Device
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    if args.multi_gpu: assert device.type == 'cuda'
    
    # Paths
    checkpoint_path = join('results', args.experiment_name, 'checkpoint')
    sample_path = join('results', args.experiment_name, 'sample')
    summary_path = join('results', args.experiment_name, 'summary')
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)
    with open(join('results', args.experiment_name, 'setting.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    writer = SummaryWriter(summary_path)
    
    # Data
    if args.dataset == 'COCO-Stuff':
        from data import COCO_Stuff
        train_dset = COCO_Stuff(args.data, mode='train')
        val_dset = COCO_Stuff(args.data, mode='val')
        n_classes = COCO_Stuff.n_classes
    train_data = data.DataLoader(train_dset, batch_size = args.batch_size, shuffle=True, drop_last=True)
    val_data = data.DataLoader(val_dset, batch_size = args.num_samples, shuffle=False, drop_last=False)
    fixed_reals, fixed_annos = next(iter(val_data))
    fixed_reals, fixed_annos = fixed_reals.to(device), fixed_annos.to(device)
    fixed_annos_onehot = onehot2d(fixed_annos, n_classes).type_as(fixed_reals)
    del val_dset
    del val_data
    vutils.save_image(fixed_reals, join(sample_path, '{:03d}_real.jpg'.format(0)), nrow=4, padding=0, normalize=True, range=(-1., 1.))
    vutils.save_image(fixed_annos.float()/n_classes, join(sample_path, '{:03d}_anno.jpg'.format(0)), nrow=4, padding=0)
    
    # Models
    E = Encoder().to(device)
    E.apply(init_weights)
    # summary(E, (3, 256, 256), device=device)
    G = Generator(n_classes).to(device)
    G.apply(init_weights)
    # summary(G, [(256,), (10, 256, 256)], device=device)
    D = Discriminator(n_classes).to(device)
    D.apply(init_weights)
    # summary(D, (13, 256, 256), device=device)
    vgg = VGG().to(device)
    
    if args.multi_gpu:
        E = nn.DataParallel(E)
        G = nn.DataParallel(G)
        # G = convert_model(G)
        D = nn.DataParallel(D)
        VGG = nn.DataParallel(VGG)
    
    # Optimizers
    G_opt = optim.Adam(itertools.chain(G.parameters(), E.parameters()), lr=args.lr_G, betas=(args.beta1, args.beta2))
    D_opt = optim.Adam(D.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
    
    # Load weights from a specific epoch
    start_ep = 0
    if args.load_epoch is not None:
        if args.load_from_experiment is None:
            load_checkpoint_path = checkpoint_path
        else:
            load_checkpoint_path = join('results', args.load_from_experiment, 'checkpoint')
        load_ep = args.load_epoch
        start_ep = load_ep + 1
        E.load_state_dict(torch.load(join(load_checkpoint_path, '{:03}.E.pth'.format(load_ep))))
        G.load_state_dict(torch.load(join(load_checkpoint_path, '{:03}.G.pth'.format(load_ep))))
        D.load_state_dict(torch.load(join(load_checkpoint_path, '{:03}.D.pth'.format(load_ep))))
        G_opt.load_state_dict(torch.load(join(load_checkpoint_path, '{:03}.G_opt.pth'.format(load_ep))))
        D_opt.load_state_dict(torch.load(join(load_checkpoint_path, '{:03}.D_opt.pth'.format(load_ep))))
    
    # Criterion
    l1_norm = nn.L1Loss()
    
    it = 0
    decayed_lr_G = args.lr_G
    decayed_lr_D = args.lr_D
    total_epochs = args.epochs + args.epochs_decay
    for ep in range(start_ep, total_epochs):
        # Linearly decay learning rates
        if ep >= args.epochs:
            decayed_lr_G = args.lr_G / args.epochs_decay * (total_epochs - ep)
            decayed_lr_D = args.lr_D / args.epochs_decay * (total_epochs - ep)
            set_lr(G_opt, decayed_lr_G)
            set_lr(D_opt, decayed_lr_D)
        # Optimize parameters
        E.train()
        G.train()
        D.train()
        for reals, annos in tqdm(train_data):
            reals, annos = reals.to(device), annos.to(device)
            annos_onehot = onehot2d(annos, n_classes).type_as(reals)
            # Train D
            trainable(E, False)
            trainable(G, False)
            trainable(D, True)
            mu, logvar = E(reals)
            latents = sample_latent(mu, logvar).detach()
            fakes = G(latents, annos_onehot).detach()
            d_real = D(reals, annos_onehot)
            d_fake = D(fakes, annos_onehot)
            # Real/fake hinge loss
            df_loss = torch.nn.ReLU()(1.0 - d_real[-1]).mean() + torch.nn.ReLU()(1.0 + d_fake[-1]).mean()
            # D loss
            d_loss = df_loss
            # Update D
            D_opt.zero_grad()
            d_loss.backward()
            D_opt.step()
            
            # Train G
            trainable(E, True)
            trainable(G, True)
            trainable(D, False)
            mu, logvar = E(reals)
            latents = sample_latent(mu, logvar)
            fakes = G(latents, annos_onehot)
            d_fake = D(fakes, annos_onehot)
            # Real/fake hinge loss
            gf_loss = -d_fake[-1].mean()
            # Feature matching loss
            fm_loss = 0
            for d_f, d_r in zip(d_fake[:-1], d_real[:-1]):
                fm_loss += l1_norm(d_f, d_r.detach())
            # Perceptual loss
            vgg_loss = 0
            for w, f, r in zip(vgg.weights, vgg(fakes), vgg(reals)):
                vgg_loss += w * l1_norm(f, r.detach())
            # KL divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
            # G loss
            g_loss = gf_loss + args.lambda_fm * (fm_loss + vgg_loss) + args.lambda_kl * kl_loss
            # Update G
            G_opt.zero_grad()
            g_loss.backward()
            G_opt.step()
            
            if (it+1) % args.log_iters == 0:
                print('iter {:d} epoch {:d} d_loss {:.4f} g_loss {:.4f} gf {:.4f} fm {:.4f} vgg_loss {:.4f} kl {:.4f}'.format(
                    it, ep, d_loss.item(), g_loss.item(), gf_loss.item(), fm_loss.item(),
                    vgg_loss.item() if type(vgg_loss) is torch.Tensor else vgg_loss, kl_loss.item()
                ))
                add_scalar_dict(writer, {
                    'd_loss': d_loss.item(),
                    'df_loss': df_loss.item()
                }, it, 'D')
                add_scalar_dict(writer, {
                    'g_loss': g_loss.item(),
                    'gf_loss': gf_loss.item(),
                    'fm_loss': fm_loss.item(),
                    'vgg_loss': vgg_loss.item() if type(vgg_loss) is torch.Tensor else vgg_loss,
                    'kl_loss': kl_loss.item()
                }, it, 'G')
                add_scalar_dict(writer, {
                    'lr_G': decayed_lr_G,
                    'lr_D': decayed_lr_D
                }, it, 'LR')
                E.eval()
                G.eval()
                with torch.no_grad():
                    mu, logvar = E(fixed_reals)
                    latents = sample_latent(mu, logvar)
                    samples = G(latents, fixed_annos_onehot)
                    vutils.save_image(samples, join(sample_path, '{:03d}_{:07d}_fake.jpg'.format(ep, it)), nrow=4, padding=0, normalize=True, range=(-1., 1.))
            it += 1
        
        # Sample images
        if (ep+1) % args.sample_epochs == 0:
            E.eval()
            G.eval()
            with torch.no_grad():
                mu, logvar = E(fixed_reals)
                latents = sample_latent(mu, logvar)
                samples = G(latents, fixed_annos_onehot)
                vutils.save_image(samples, join(sample_path, '{:03d}_fake.jpg'.format(ep)), nrow=4, padding=0, normalize=True, range=(-1., 1.))
        
        # Checkpoints
        if (ep+1) % args.save_epochs == 0:
            torch.save(E.state_dict(), join(checkpoint_path, '{:03}.E.pth'.format(ep)))
            torch.save(G.state_dict(), join(checkpoint_path, '{:03}.G.pth'.format(ep)))
            torch.save(D.state_dict(), join(checkpoint_path, '{:03}.D.pth'.format(ep)))
            torch.save(G_opt.state_dict(), join(checkpoint_path, '{:03}.G_opt.pth'.format(ep)))
            torch.save(D_opt.state_dict(), join(checkpoint_path, '{:03}.D_opt.pth'.format(ep)))