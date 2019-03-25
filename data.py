# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Data loaders"""

import numpy as np
from glob import glob
from os.path import basename, join
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


def crop_and_resize(img1, img2, center=False):
    assert img1.size == img2.size, 'Error: Image sizes {}, {} are not the same.'.format(img.size, ann.size)
    w, h = img1.size
    if w != h:
        if w < h:  # width < height
            if center:
                x, y = 0, (h-w)//2
            else:
                x, y = 0, np.random.randint(0, h-w+1)
            w, h = w, w
        else:      # width > height
            if center:
                x, y = (w-h)//2, 0
            else:
                x, y = np.random.randint(0, w-h+1), 0
            w, h = h, h
        img1 = img1.crop((x, y, x+w, y+h))
        img2 = img2.crop((x, y, x+w, y+h))
    w, h = 256, 256
    img1 = img1.resize((w, h), Image.NEAREST)
    img2 = img2.resize((w, h), Image.NEAREST)
    return img1, img2

class COCO_Stuff(data.Dataset):
    n_classes = 183  # 0-181; 255 is unlabelled
    def __init__(self, root, mode='train'):
        if mode == 'train':
            imgs = glob(join(root, 'images/train2017/*.jpg'))
            anns = glob(join(root, 'annotations/train2017/*.png'))
            self.center_crop = False
        if mode == 'val':
            imgs = glob(join(root, 'images/val2017/*.jpg'))
            anns = glob(join(root, 'annotations/val2017/*.png'))
            self.center_crop = True
        assert len(imgs) == len(anns), 'Error: Got different numbers of images and annotations.'
        self.imgs = sorted(imgs, key=lambda x: int(basename(x).split('.')[0]))
        self.anns = sorted(anns, key=lambda x: int(basename(x).split('.')[0]))
        
        # Check file names are matched
        for img, ann in zip(self.imgs, self.anns):
            if basename(img).rsplit('.', 1)[0] != basename(ann).rsplit('.', 1)[0]:
                raise Exception('Image and annotation files are not matched: {}, {}'.format(img, ann))
    
    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        ann = Image.open(self.anns[index])
        img, ann = crop_and_resize(img, ann, self.center_crop)
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
        return img, ann
    
    def __len__(self):
        return len(self.imgs)