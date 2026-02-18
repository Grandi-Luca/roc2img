import numpy as np
import torch
from sklearn.neighbors import KernelDensity

from utils import ResNetModel


MU_WEIGHTS = 0
STD_WEIGHTS_RESNET_50 = 0.3
STD_WEIGHTS_RESNET_18 = 0.13

MU_BIAS_RESNET_50 = -0.32
STD_BIAS_RESNET_50 = 1.59

MU_BIAS_RESNET_18 = -0.19
STD_BIAS_RESNET_18 = 0.20

MU_BIAS_RESNET_101 = -0.09
STD_BIAS_RESNET_101 = 1.14

MU_BIAS_RESNET_152 = -0.13
STD_BIAS_RESNET_152 = 0.97


KDE_MODELS: dict = {
    'weight': None,
    'bias': None,
}


def extract_weights_and_biases(model: ResNetModel) -> tuple[np.ndarray, np.ndarray]:
    # First convolution layer weights
    weights = model.get_state_dict(
    )['conv1.weight'].detach().cpu().numpy().flatten()

    # Global bias from all batch norm layers
    bias_list = []
    for name, param in model.get_state_dict().items():
        if "bn" in name and "bias" in name:
            bias_list.append(param.detach().cpu().numpy().flatten())

    return weights, np.concatenate(bias_list)


def fit_kde_models(weights, bias) -> None:
    global KDE_MODELS
    KDE_MODELS['weight'] = KernelDensity(
        kernel='gaussian', bandwidth=0.5).fit(weights.reshape(-1, 1))
    KDE_MODELS['bias'] = KernelDensity(
        kernel='gaussian', bandwidth=0.5).fit(bias.reshape(-1, 1))


#  ##### GAUSS DISTRIBUTIONS FOR RESNET WIGHT (FIRST) #####
def _approx_resnet_50_weight_first_gauss_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        MU_WEIGHTS, STD_WEIGHTS_RESNET_50, (cout, cin//groups, kd, kh, kw)).astype(np.float32)
    samples = samples - samples.mean(axis=(2, 3, 4), keepdims=True)
    return torch.from_numpy(samples)


def _approx_resnet_18_weight_first_gauss_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        MU_WEIGHTS, STD_WEIGHTS_RESNET_18, (cout, cin//groups, kd, kh, kw)).astype(np.float32)
    samples = samples - samples.mean(axis=(2, 3, 4), keepdims=True)
    return torch.from_numpy(samples)


#  ##### LAPLACE DISTRIBUTIONS FOR RESNET WIGHT (FIRST) #####
def _approx_resnet_50_weight_first_laplace_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.laplace(
        MU_WEIGHTS, STD_WEIGHTS_RESNET_50 / np.sqrt(2), (cout, cin//groups, kd, kh, kw)).astype(np.float32)
    samples = samples - samples.mean(axis=(2, 3, 4), keepdims=True)
    return torch.from_numpy(samples)


def _approx_resnet_18_weight_first_laplace_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.laplace(
        MU_WEIGHTS, STD_WEIGHTS_RESNET_18 / np.sqrt(2), (cout, cin//groups, kd, kh, kw)).astype(np.float32)
    samples = samples - samples.mean(axis=(2, 3, 4), keepdims=True)
    return torch.from_numpy(samples)


def _gaussian_01_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        0, 1, (cout, cin//groups, kd, kh, kw)).astype(np.float32)
    samples = samples - samples.mean(axis=(2, 3, 4), keepdims=True)
    return torch.from_numpy(samples)


def _conv2d_weight_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    assert groups > 0, "Groups must be a positive integer."
    assert cin % groups == 0, "cin must be divisible by groups."
    k = groups / (cin * kd * kh * kw)
    samples = np.random.uniform(-np.sqrt(k), np.sqrt(k),
                                size=(cout, cin//groups, kd, kh, kw)).astype(np.float32)
    return torch.from_numpy(samples)


def _real_resnet_weight_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    global KDE_MODELS
    assert KDE_MODELS['weight'] is not None, "KDE model for weights is not fitted yet. Extract the weights and then fit KDE model."
    samples = KDE_MODELS['weight'].sample(cout * (cin // groups) * kd * kh * kw)
    samples = samples.reshape(cout, cin // groups, kd, kh, kw).astype(np.float32)
    samples = add_random_noise(torch.from_numpy(samples), noise_level=0)
    samples = samples - samples.mean(axis=(2, 3, 4), keepdims=True)
    return samples


def _conv2d_bias_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    assert groups > 0, "Groups must be a positive integer."
    assert cin % groups == 0, "cin must be divisible by groups."
    k = groups / (cin * kd * kh * kw)
    samples = np.random.uniform(-np.sqrt(k), np.sqrt(k),
                                size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _uniform_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.uniform(
        -1, 1, size=cout).astype(np.float32)
    return torch.from_numpy(samples)

def add_random_noise(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    mask = torch.rand_like(tensor) < 0.2 
    noise = torch.randn_like(tensor) * noise_level
    noise = noise * mask.float()
    return tensor + noise

#  ##### GAUSS DISTRIBUTIONS FOR RESNET BIASES (GLOBAL) #####
def _approx_resnet18_bias_global_gauss_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        MU_BIAS_RESNET_18, STD_BIAS_RESNET_18, size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _approx_resnet50_bias_global_gauss_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        MU_BIAS_RESNET_50, STD_BIAS_RESNET_50, size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _approx_resnet101_bias_global_gauss_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        MU_BIAS_RESNET_101, STD_BIAS_RESNET_101, size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _approx_resnet152_bias_global_gauss_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.normal(
        MU_BIAS_RESNET_152, STD_BIAS_RESNET_152, size=cout).astype(np.float32)
    return torch.from_numpy(samples)

# ##### LAPLACE DISTRIBUTIONS FOR RESNET BIASES (GLOBAL) #####
def _approx_resnet18_bias_global_laplace_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.laplace(
        MU_BIAS_RESNET_18, STD_BIAS_RESNET_18/np.sqrt(2), size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _approx_resnet50_bias_global_laplace_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.laplace(
        MU_BIAS_RESNET_50, STD_BIAS_RESNET_50/np.sqrt(2), size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _approx_resnet101_bias_global_laplace_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.laplace(
        MU_BIAS_RESNET_101, STD_BIAS_RESNET_101/np.sqrt(2), size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _approx_resnet152_bias_global_laplace_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    samples = np.random.laplace(
        MU_BIAS_RESNET_152, STD_BIAS_RESNET_152/np.sqrt(2), size=cout).astype(np.float32)
    return torch.from_numpy(samples)


def _real_resnet_bias_distr(cout: int, cin: int, kd: int, kh: int, kw: int, groups: int = 1) -> torch.Tensor:
    global KDE_MODELS
    assert KDE_MODELS['bias'] is not None, "KDE model for biases is not fitted yet."
    samples = KDE_MODELS['bias'].sample(cout)
    samples = samples.reshape(cout).astype(np.float32)
    return torch.from_numpy(samples)


class Distribution():
    def __init__(self, name, func):
        self.name = name
        self.fn = func


class DistributionType():
    # WEIGHT DISTRIBUTIONS
    APPROX_RESNET50_WEIGHT_FIRST_GAUSS = Distribution(
        "APPROX_RESNET50_FIRST_GAUSS", _approx_resnet_50_weight_first_gauss_distr)
    APPROX_RESNET18_WEIGHT_FIRST_GAUSS = Distribution(
        "APPROX_RESNET18_FIRST_GAUSS", _approx_resnet_18_weight_first_gauss_distr)

    APPROX_RESNET50_WEIGHT_FIRST_LAPLACE = Distribution(
        "APPROX_RESNET50_FIRST_LAPLACE", _approx_resnet_50_weight_first_laplace_distr)
    APPROX_RESNET18_WEIGHT_FIRST_LAPLACE = Distribution(
        "APPROX_RESNET18_FIRST_LAPLACE", _approx_resnet_18_weight_first_laplace_distr)

    APPROX_RESNET101_WEIGHT_FIRST_LAPLACE = Distribution(
        "APPROX_RESNET101_FIRST_LAPLACE", _approx_resnet_50_weight_first_laplace_distr)
    APPROX_RESNET101_WEIGHT_FIRST_GAUSS = Distribution(
        "APPROX_RESNET101_FIRST_GAUSS", _approx_resnet_50_weight_first_gauss_distr)

    APPROX_RESNET152_WEIGHT_FIRST_LAPLACE = Distribution(
        "APPROX_RESNET152_FIRST_LAPLACE", _approx_resnet_50_weight_first_laplace_distr)
    APPROX_RESNET152_WEIGHT_FIRST_GAUSS = Distribution(
        "APPROX_RESNET152_FIRST_GAUSS", _approx_resnet_50_weight_first_gauss_distr)

    GAUSSIAN_01 = Distribution("GAUSSIAN_01", _gaussian_01_distr)

    CONV2D_WEIGHT = Distribution("CONV2D_WEIGHT", _conv2d_weight_distr)

    REAL_RESNET18_WEIGHT = Distribution(
        "REAL_RESNET18", _real_resnet_weight_distr)
    REAL_RESNET50_WEIGHT = Distribution(
        "REAL_RESNET50", _real_resnet_weight_distr)
    REAL_RESNET101_WEIGHT = Distribution(
        "REAL_RESNET101", _real_resnet_weight_distr)
    REAL_RESNET152_WEIGHT = Distribution(
        "REAL_RESNET152", _real_resnet_weight_distr)

    # BIAS DISTRIBUTIONS
    CONV2D_BIAS = Distribution("CONV2D_BIAS", _conv2d_bias_distr)

    UNIFORM = Distribution("UNIFORM", _uniform_distr)

    REAL_RESNET18_BIAS = Distribution(
        "REAL_RESNET18", _real_resnet_bias_distr)
    REAL_RESNET50_BIAS = Distribution(
        "REAL_RESNET50", _real_resnet_bias_distr)
    REAL_RESNET101_BIAS = Distribution(
        "REAL_RESNET101", _real_resnet_bias_distr)
    REAL_RESNET152_BIAS = Distribution(
        "REAL_RESNET152", _real_resnet_bias_distr)

    APPROX_RESNET18_BIAS_GLOBAL_GAUSS = Distribution(
        "APPROX_RESNET18_GLOBAL_GAUSS", _approx_resnet18_bias_global_gauss_distr)
    APPROX_RESNET101_BIAS_GLOBAL_GAUSS = Distribution(
        "APPROX_RESNET101_GLOBAL_GAUSS", _approx_resnet101_bias_global_gauss_distr)
    APPROX_RESNET152_BIAS_GLOBAL_GAUSS = Distribution(
        "APPROX_RESNET152_GLOBAL_GAUSS", _approx_resnet152_bias_global_gauss_distr)
    APPROX_RESNET50_BIAS_GLOBAL_GAUSS = Distribution(
        "APPROX_RESNET50_GLOBAL_GAUSS", _approx_resnet50_bias_global_gauss_distr)

    APPROX_RESNET18_BIAS_GLOBAL_LAPLACE = Distribution(
        "APPROX_RESNET18_GLOBAL_LAPLACE", _approx_resnet18_bias_global_laplace_distr)
    APPROX_RESNET101_BIAS_GLOBAL_LAPLACE = Distribution(
        "APPROX_RESNET101_GLOBAL_LAPLACE", _approx_resnet101_bias_global_laplace_distr)
    APPROX_RESNET152_BIAS_GLOBAL_LAPLACE = Distribution(
        "APPROX_RESNET152_GLOBAL_LAPLACE", _approx_resnet152_bias_global_laplace_distr)
    APPROX_RESNET50_BIAS_GLOBAL_LAPLACE = Distribution(
        "APPROX_RESNET50_GLOBAL_LAPLACE", _approx_resnet50_bias_global_laplace_distr)
