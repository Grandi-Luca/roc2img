import torch
import numpy as np
from torchvision import models

import random
import os


class PaddingMode():
    ON = True
    OFF = False
    RANDOM = 'random'


class ConvolutionType():
    STANDARD = 'standard'
    SPATIAL = 'spatial'
    DEPTHWISE_SEP = 'depthwise_sep'
    DEPTHWISE = 'depthwise'


class FeatureType():
    PPV = 'ppv'
    MPV = 'mpv'
    LSPV = 'lspv'
    MIPV = 'mipv'
    MAXPV = 'maxpv'
    MINPV = 'minpv'
    GMPV = 'gmpv'
    ENTROPY = 'entropy'
    MAX2D = 'max2d'
    AVG2D = 'avg2d'


class DilationType():
    DILATED_1 = 1
    DILATED_2 = 2
    DILATED_3 = 3
    RANDOM_13 = 'random_1_3'
    UNIFORM_ROCKET = 'uniform_rocket'
    RANDOM_02 = 'random_0_2'
    RANDOM_03 = 'random_0_3'
    RANDOM_04 = 'random_0_4'


class ResNetModel():
    RESNET18 = models.ResNet18_Weights.DEFAULT
    RESNET50 = models.ResNet50_Weights.DEFAULT
    RESNET101 = models.ResNet101_Weights.DEFAULT
    RESNET152 = models.ResNet152_Weights.DEFAULT


def _set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
