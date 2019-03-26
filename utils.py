# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Common utilities"""

import torch


def onehot2d(x, n):
    assert x.dim() == 4 and x.size(1) == 1
    return torch.zeros_like(x).repeat(1, n, 1, 1).scatter_(1, x, 1)