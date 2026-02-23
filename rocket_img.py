import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from collections.abc import Callable

from typing import Optional

from utils import ConvolutionType, PaddingMode, FeatureType, DilationType
from utils import _set_random_seed, ResNetModel
from distributions import DistributionType, extract_weights_and_biases, fit_kde_models


def _sample_rocket_dilation(input_dim: int, kernel_size: int, num: int, dilation_type: DilationType) -> np.ndarray:
    """Sample random dilations for ROCKET kernels using vectorized operations."""
    if kernel_size <= 1 or input_dim <= kernel_size or dilation_type == DilationType.DILATED_1:
        return np.ones(num, dtype=np.int32)
    
    if dilation_type == DilationType.UNIFORM_ROCKET:        
        max_dilation = (input_dim - 1) / float(kernel_size - 1)
        upper = float(np.log2(max_dilation))
        return np.int32(2 ** np.random.uniform(0.0, upper, size=num))
    
    elif dilation_type == DilationType.DILATED_2:
        return np.full(num, 2, dtype=np.int32)
    elif dilation_type == DilationType.DILATED_3:
        return np.full(num, 3, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported dilation type: {dilation_type}")


def _generate_kernels(
    cout: int,
    cin: int,
    input_dhw: tuple[int, int, int] | int,
    candidate_lengths: list,
    padding_mode: PaddingMode,
    weight_distr_fn: Callable,
    bias_distr_fn: Callable,
    dilation_type: DilationType,
    candidate_strides: list[int],
    device: Optional[torch.device] = None,
    random_state: Optional[int] = None
) -> dict:

    if random_state is not None:
        _set_random_seed(random_state)

    if isinstance(input_dhw, int):
        input_depth, input_h, input_w = input_dhw, input_dhw, input_dhw
    else:
        input_depth, input_h, input_w = input_dhw

    # Generate padding mask
    candidate_pad = {
        PaddingMode.ON: np.ones(cout, dtype=bool),
        PaddingMode.OFF: np.zeros(cout, dtype=bool),
        PaddingMode.RANDOM: np.random.randint(2, size=cout, dtype=bool)
    }[padding_mode]

    # Generate random kernel parameters
    kernel_sizes = np.random.choice(candidate_lengths, cout)
    strides = np.random.choice(candidate_strides, cout)

    # Initialize arrays for dilations
    dilations_h = np.ones(cout, dtype=np.int32)
    dilations_w = np.ones(cout, dtype=np.int32)
    dilations_d = np.ones(cout, dtype=np.int32)
    
    # Compute dilations for each unique size
    for size in candidate_lengths:
        mask = kernel_sizes == size
        num_ker = mask.sum()
        if num_ker == 0:
            continue

        input_dim = min(input_depth, input_h, input_w) if input_depth > 1 else min(input_h, input_w)
        d = _sample_rocket_dilation(input_dim, size, num_ker, dilation_type)
        dilations_h[mask] = d
        dilations_w[mask] = d
        dilations_d[mask] = d if input_depth > 1 else 1

    # Compute paddings vectorized
    paddings_d = np.where(candidate_pad, ((kernel_sizes - 1) * dilations_d) // 2, 0) if input_depth > 1 else np.zeros(cout, dtype=np.int32)
    paddings_h = np.where(candidate_pad, ((kernel_sizes - 1) * dilations_h) // 2, 0)
    paddings_w = np.where(candidate_pad, ((kernel_sizes - 1) * dilations_w) // 2, 0)

    # Group kernels by unique parameter combinations
    kernel_params = np.column_stack([kernel_sizes, dilations_d, dilations_h, dilations_w, paddings_d, paddings_h, paddings_w, strides])
    unique_params, counts = np.unique(kernel_params, axis=0, return_counts=True)

    # Generate weights and biases for each unique parameter group
    groups = {}
    for key, count in zip(unique_params, counts):
        kernel_size = key[0]
        depth = kernel_size if input_depth > 1 else 1
        # Generate biases
        bias = bias_distr_fn(count, cin, depth, kernel_size, kernel_size).to(device)
  
        weights = (
            weight_distr_fn(count, cin, depth, kernel_size, kernel_size).to(device),
        )

        groups[tuple(key.tolist())] = {"weights": weights, "bias": bias}

    return groups


class ROCKET(nn.Module):
    def __init__(
            self,
            cout: int,
            cin: int,
            input_dhw: tuple[int, int, int] | int,
            candidate_lengths: list[int],
            features_to_extract: list[FeatureType],
            distr_pair: tuple[DistributionType, DistributionType] = (
                DistributionType.GAUSSIAN_01, DistributionType.UNIFORM),
            candidate_strides: list[int] = [1],
            dilation: DilationType = DilationType.UNIFORM_ROCKET,
            padding_mode: PaddingMode = PaddingMode.RANDOM,
            device: Optional[torch.device] = None,
            random_state=None):

        super().__init__()

        self.cout = cout
        self.candidate_lengths = candidate_lengths
        self.padding_mode = padding_mode
        self.candidate_strides = candidate_strides
        self.dilation = dilation
        self.distr_pair = distr_pair

        self.device = device
        self.training = False

        if len(features_to_extract) == 0:
            raise ValueError("At least one feature must be specified.")
        self.features_to_extract = list(set(features_to_extract))

        self.random_state = random_state if isinstance(
            random_state, int) else None

        # Extract weights and biases from ResNet model and fit KDE models
        model = None
        if self.distr_pair[0] == DistributionType.REAL_RESNET18_WEIGHT:
            model = ResNetModel.RESNET18
        elif self.distr_pair[0] == DistributionType.REAL_RESNET50_WEIGHT:
            model = ResNetModel.RESNET50
        elif self.distr_pair[0] == DistributionType.REAL_RESNET101_WEIGHT:
            model = ResNetModel.RESNET101
        elif self.distr_pair[0] == DistributionType.REAL_RESNET152_WEIGHT:
            model = ResNetModel.RESNET152

        if model is not None:
            w, b = extract_weights_and_biases(model)
            fit_kde_models(w, b)

        # Generate random convolutional kernels
        self.conv_params = _generate_kernels(
            cout=self.cout,
            cin=cin,
            input_dhw=input_dhw,
            candidate_lengths=self.candidate_lengths,
            padding_mode=self.padding_mode,
            weight_distr_fn=self.distr_pair[0].fn,
            bias_distr_fn=self.distr_pair[1].fn,
            dilation_type=self.dilation,
            candidate_strides=self.candidate_strides,
            device=self.device,
            random_state=self.random_state
        )

    def forward(self, X):
        """Forward pass through the ROCKET transformation.
        
        Args:
            X (torch.Tensor): Input data of shape [B, C, H, W].
            
        Returns:
            torch.Tensor: Transformed features.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        
        return self._transform(X)

    def _transform(self, X):
        batch_size, _, _, _, _ = X.shape

        output_size = len(self.features_to_extract) * self.cout
        if FeatureType.MAX2D in self.features_to_extract:
            output_size = output_size - self.cout + 4 * self.cout
        if FeatureType.AVG2D in self.features_to_extract:
            output_size = output_size - self.cout + 4 * self.cout

        features = torch.zeros((batch_size, output_size))
        
        s = 0
        for key, var in self.conv_params.items():

            _, dilation_d, dilation_h, dilation_w, padding_d, padding_h, padding_w, stride = key

            padding_1 = (padding_d, padding_h, padding_w)
            dilation_1 = (dilation_d, dilation_h, dilation_w)
            group = 1

            # First convolution
            data = F.conv3d(
                input=X,
                weight=var['weights'][0],
                bias=var['bias'],
                stride=stride,
                padding=padding_1,
                dilation=dilation_1,
                groups=group
            ).relu()

            features_list = []
            # Computing Max pooling
            if FeatureType.MAX2D in self.features_to_extract:
                max2d = F.adaptive_max_pool3d(data, output_size=(1,2,2)).flatten(1)
                features_list.append(max2d)

            if FeatureType.AVG2D in self.features_to_extract:
                avg2d = F.adaptive_avg_pool3d(data, output_size=(1,2,2)).flatten(1)
                features_list.append(avg2d)

            # [B, num_kernel_group, H*W]
            data = data.flatten(2)
            N = data.shape[-1]
            # Positive indices
            pos_mask = data > 0
            pos_count = pos_mask.sum(-1)

            if FeatureType.MPV in self.features_to_extract or FeatureType.MIPV in self.features_to_extract:
                zero_mask = pos_count == 0
                pos_count_clone = pos_count.to(torch.float32)
                pos_count_clone[zero_mask] = 1  # to avoid division by zero

            # Extract features
            for feature in self.features_to_extract:

                # Computing PPV
                if feature == FeatureType.PPV:
                    ppv = pos_count.to(torch.float32) / float(N)
                    features_list.append(ppv)
                    del ppv

                # Computing MPV
                elif feature == FeatureType.MPV:
                    mpv_num = (data*pos_mask).sum(-1)
                    mpv = mpv_num / pos_count_clone
                    # If no positive, set to 0
                    mpv[zero_mask] = 0
                    features_list.append(mpv)
                    del mpv_num, mpv

                # Computing MIPV
                elif feature == FeatureType.MIPV:
                    idx = torch.arange(N, dtype=torch.float32, device=data.device)
                    mipv_num = pos_mask.to(torch.float32) @ idx   # [B, K]
                    mipv = mipv_num / pos_count_clone
                    mipv[zero_mask] = -1.0
                    features_list.append(mipv)
                    del mipv_num, mipv

                # Computing LSPV
                elif feature == FeatureType.LSPV:
                    pos_int = pos_mask.to(torch.int32)          # [B, K, N]
                    # cumsum of True
                    cumsum_pos = pos_int.cumsum(dim=-1)         # [B, K, N]
                    # cumsum values at positions of False
                    zero_cumsum = (~pos_mask).to(torch.int32) * cumsum_pos
                    # Last cumsum value before each position
                    last_zero_cumsum = torch.cummax(zero_cumsum, dim=-1).values

                    # current run length = cumsum - last_zero_cumsum
                    run_lengths = cumsum_pos - last_zero_cumsum  # [B, K, N]

                    # LSPV = max run along N
                    lspv = run_lengths.max(dim=-1).values       # [B, K]
                    features_list.append(lspv)
                    del pos_int, cumsum_pos, zero_cumsum, last_zero_cumsum, run_lengths, lspv

                # Computing Global Max Pooling
                elif feature == FeatureType.MAXPV:
                    features_list.append(data.max(-1).values)

                # Computing Global Min Pooling
                elif feature == FeatureType.MINPV:
                    features_list.append(data.min(-1).values)


                # Computing Generalized Mean Pooling
                elif feature == FeatureType.GMPV:
                    p = 2.0
                    gmpv = ((data**p).mean(-1)) ** (1/p)
                    features_list.append(gmpv)
                    del gmpv

                elif feature == FeatureType.ENTROPY:
                    eps = 1e-12
                    p = torch.softmax(data, dim=-1)
                    entropy = - (p * (p + eps).log()).sum(dim=-1)
                    features_list.append(entropy)

            extracted_features = torch.cat(features_list, dim=1)
            e = s + extracted_features.shape[1]
            features[:, s:e] = extracted_features
            s = e
            del data, features_list, extracted_features

        return features

    def get_params(self) -> dict[str, any]:
        return {
            "cout": self.cout,
            "candidate_lengths": self.candidate_lengths,
            "padding_mode": self.padding_mode,
            "dilation": self.dilation,
            "candidate_strides": self.candidate_strides,
            "convolution_type": 'standard',
            "features_to_extract": sorted(self.features_to_extract),
            "distr_pair": (self.distr_pair[0].name, self.distr_pair[1].name),
            "random_state": self.random_state,
            'device': str(self.device),
        }