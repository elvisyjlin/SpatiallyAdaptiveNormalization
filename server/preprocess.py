# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Preprocess demo datasets"""

import json
import os
import numpy as np
import sys
from glob import glob
from os.path import basename, join
from PIL import Image

from .segmentation import get_colormap_values, encode_image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import crop_and_resize


def preprocess_demo_images(dataset, path, num=-1, ids=[]):
    print('Preprocessing', dataset, '...')
    demo_path = join(path, 'demo_images')
    os.makedirs(demo_path, exist_ok=True)
    if dataset.lower() == 'cocostuff':
        colormap = get_colormap_values()
        imgs = glob(join(path, 'images/val2017/*.jpg'))
        imgs = sorted(imgs, key=lambda x: int(basename(x).split('.')[0]))
        anns = glob(join(path, 'annotations/val2017/*.png'))
        anns = sorted(anns, key=lambda x: int(basename(x).split('.')[0]))
        assert len(imgs) == len(anns)
        
        label_list = {}
        with open(join(path, 'labels.txt'), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                index, label = line.strip().split(': ')
                if index == '0':
                    label_list[255] = label
                else:
                    label_list[int(index)-1] = label
        with open(join(demo_path, 'class_list.json'), 'w', encoding='utf-8') as f:
            json.dump(label_list, f)
        
        for idx, (img, ann) in enumerate(zip(imgs, anns)):
            if idx < num or idx in ids:
                img = Image.open(img)
                ann = Image.open(ann)
                img, ann = crop_and_resize(img, ann, True)
                lab = np.unique(np.array(ann))
                ann = encode_image(ann, colormap)
                img.save(join(demo_path, '{:d}_img.png'.format(idx)))
                ann.save(join(demo_path, '{:d}_seg.png'.format(idx)))
                np.savez(join(demo_path, '{:d}_lab.npz'.format(idx)), lab=lab)
        return idx+1
    raise ValueError('Not supported dataset ' + dataset)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['cocostuff'])
    parser.add_argument('path', type=str)
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--ids', type=int, nargs='*', default=[])
    args = parser.parse_args()
    assert not(args.num > 0 and len(args.ids) > 0), 'Error: Either --num or --ids can be specified at one time.'
    preprocess_demo_images(args.dataset, args.path, args.num, args.ids)