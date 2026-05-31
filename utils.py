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
    GAP = 'gap'


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


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, p: int) -> None:
        super().__init__()
        self.p: int = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.flatten(2)**self.p).mean(-1)**(1/self.p)

    def get_output_size(self) -> int:
        return (1,)

class AGeM(nn.Module):
    def __init__(self, p: int, output_size: int | tuple[int, int, int]) -> None:
        super().__init__()

        assert p > 0, "p must be greater than 0"
        self.p: int = p
        self.output_size: tuple[int, int, int] = output_size if isinstance(output_size, tuple) else (output_size, output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.adaptive_avg_pool3d(x.pow(self.p), self.output_size).pow(1.0 / self.p)
        return out

    def get_output_size(self) -> int | tuple[int, int, int]:
        return self.output_size

class AAP(nn.Module):
    def __init__(self, output_size: int | tuple[int, int, int]) -> None:
        super().__init__()
        self.output_size: tuple[int, int, int] = output_size if isinstance(output_size, tuple) else (output_size, output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.adaptive_avg_pool3d(x, self.output_size)
        return out

    def get_output_size(self) -> int | tuple[int, int, int]:
        return self.output_size

class AMP(nn.Module):
    def __init__(self, output_size: int | tuple[int, int, int]) -> None:
        super().__init__()
        self.output_size: tuple[int, int, int] = output_size if isinstance(output_size, tuple) else (output_size, output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.adaptive_max_pool3d(x, self.output_size)
        return out

    def get_output_size(self) -> int | tuple[int, int, int]:
        return self.output_size
