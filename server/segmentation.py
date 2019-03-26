# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Helper functions for semantic segmentation"""

import colorsys
import numpy as np
from matplotlib import cm
from PIL import Image, PngImagePlugin


def get_colormap_values(fn=cm.gist_ncar, shuffle_seed=1234):
    colormap = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap[i, 0] = np.int_(np.round(fn(i)[0] * 255.0))
        colormap[i, 1] = np.int_(np.round(fn(i)[1] * 255.0))
        colormap[i, 2] = np.int_(np.round(fn(i)[2] * 255.0))
    if shuffle_seed is not None and type(shuffle_seed) is int:
        st0 = np.random.get_state()
        np.random.seed(shuffle_seed)
        np.random.shuffle(colormap)
        np.random.set_state(st0)
    return colormap

def encode_image(image, colormap=None):
    if colormap is None:
        colormap = get_colormap_values()
    if type(image) in [Image.Image, PngImagePlugin.PngImageFile]:
        image = np.array(image)
    assert image.ndim == 2
    h, w = image.shape[0], image.shape[1]
    colored_array = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        colored_array[image==i, :] = color
    return Image.fromarray(colored_array)

def decode_image(image, colormap=None):
    if colormap is None:
        colormap = get_colormap_values()
    if type(image) in [Image.Image, PngImagePlugin.PngImageFile]:
        image = np.array(image)
    assert image.ndim == 3
    if image.shape[2] == 4:
        image = image[..., :3]
    h, w = image.shape[0], image.shape[1]
    segmented_array = np.zeros((h, w), dtype=np.uint8)
    for i, color in enumerate(colormap):
        segmented_array[(image==color).all(axis=2)] = i
    return Image.fromarray(segmented_array)