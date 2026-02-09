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
    if kernel_size <= 1 or dilation_type == DilationType.DILATED_1:
        return np.ones(num, dtype=np.int32)
    
    if dilation_type == DilationType.UNIFORM_ROCKET:
        max_dilation = (input_dim - 1) / float(kernel_size - 1)
        if max_dilation <= 1.0:
            return np.ones(num, dtype=np.int32)
        
        upper = float(np.log2(max_dilation))
        return np.int32(2 ** np.random.uniform(0.0, upper, size=num))
    
    elif dilation_type == DilationType.DILATED_2:
        return np.full(num, 2, dtype=np.int32)
    elif dilation_type == DilationType.DILATED_3:
        return np.full(num, 3, dtype=np.int32)
    elif dilation_type == DilationType.RANDOM_13:
        return  np.int32(np.random.choice([1,2,3], size=num))
    elif dilation_type == DilationType.RANDOM_02:
        return  np.int32(2 ** np.random.choice([0,1], size=num))
    elif dilation_type == DilationType.RANDOM_03:
        return  np.int32(2 ** np.random.choice([0,1,2], size=num))
    elif dilation_type == DilationType.RANDOM_04:
        return  np.int32(2 ** np.random.choice([0,1,2,3], size=num))
    else:
        raise ValueError(f"Unsupported dilation type: {dilation_type}")


def _generate_kernels(
    cin: int,
    input_hw: tuple[int, int] | int,
    cout: int,
    convolution_type: ConvolutionType,
    candidate_sizes: list,
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

    if isinstance(input_hw, int):
        input_h, input_w = input_hw, input_hw
    else:
        input_h, input_w = input_hw

    # Generate padding mask
    candidate_pad = {
        PaddingMode.ON: np.ones(cout, dtype=bool),
        PaddingMode.OFF: np.zeros(cout, dtype=bool),
        PaddingMode.RANDOM: np.random.randint(2, size=cout, dtype=bool)
    }[padding_mode]

    # Generate random kernel parameters
    kernel_sizes = np.random.choice(candidate_sizes, cout)
    strides = np.random.choice(candidate_strides, cout)

    # Initialize arrays for dilations
    dilations_h = np.ones(cout, dtype=np.int32)
    dilations_w = np.ones(cout, dtype=np.int32)
    
    # Compute dilations for each unique size
    for size in candidate_sizes:
        mask = kernel_sizes == size
        num_ker = mask.sum()
        if num_ker == 0:
            continue

        if convolution_type == ConvolutionType.SPATIAL:
            dilations_h[mask] = _sample_rocket_dilation(input_h, size, num_ker, dilation_type)
            dilations_w[mask] = _sample_rocket_dilation(input_w, size, num_ker, dilation_type)
        else:
            d = _sample_rocket_dilation(min(input_h, input_w), size, num_ker, dilation_type)
            dilations_h[mask] = d
            dilations_w[mask] = d

    # Compute paddings vectorized
    paddings_h = np.where(candidate_pad, ((kernel_sizes - 1) * dilations_h) // 2, 0)
    paddings_w = np.where(candidate_pad, ((kernel_sizes - 1) * dilations_w) // 2, 0)

    # Group kernels by unique parameter combinations
    kernel_params = np.column_stack([kernel_sizes, dilations_h, dilations_w, paddings_h, paddings_w, strides])
    unique_params, counts = np.unique(kernel_params, axis=0, return_counts=True)

    # Generate weights and biases for each unique parameter group
    groups = {}
    for key, count in zip(unique_params, counts):
        kernel_size = key[0]
        
        # Generate biases
        if convolution_type == ConvolutionType.DEPTHWISE:
            bias = bias_distr_fn(count * cin, cin, kernel_size, kernel_size, groups=cin).to(device)
        else:
            bias = bias_distr_fn(count, cin, kernel_size, kernel_size).to(device)

        # Generate weights based on convolution type
        if convolution_type == ConvolutionType.SPATIAL:
            weights = (
                weight_distr_fn(count, cin, 1, kernel_size).to(device),
                weight_distr_fn(count, cin, kernel_size, 1).to(device)
            )
        elif convolution_type == ConvolutionType.DEPTHWISE_SEP:
            weights = (
                weight_distr_fn(cin, cin, kernel_size, kernel_size, groups=cin).to(device),
                weight_distr_fn(count, cin, 1, 1).to(device)
            )
        elif convolution_type == ConvolutionType.DEPTHWISE:
            weights = (weight_distr_fn(count * cin, cin, kernel_size, kernel_size, groups=cin).to(device),)
        else:  # STANDARD
            weights = (weight_distr_fn(count, cin, kernel_size, kernel_size).to(device),)

        groups[tuple(key.tolist())] = {"weights": weights, "bias": bias}

    return groups


class ROCKET(nn.Module):
    def __init__(
            self,
            cin: int,
            input_h: int,
            input_w: int,
            cout=1000,
            candidate_lengths: list[int] = [3],
            padding_mode: PaddingMode = PaddingMode.RANDOM,
            distr_pair: tuple[DistributionType, DistributionType] = (
                DistributionType.REAL_RESNET101_WEIGHT, DistributionType.REAL_RESNET101_BIAS),
            dilation: DilationType = DilationType.UNIFORM_ROCKET,
            candidate_strides: list[int] = [1],
            convolution_type: ConvolutionType = ConvolutionType.STANDARD,
            features_to_extract: list[FeatureType] = [
                FeatureType.PPV, FeatureType.MPV, FeatureType.MIPV, FeatureType.LSPV],
            device: Optional[torch.device] = None,
            random_state=None):

        super().__init__()
        
        self.cin = cin
        self.input_h = input_h
        self.input_w = input_w
        self.cout = cout
        self.candidate_lengths = candidate_lengths
        self.padding_mode = padding_mode
        self.candidate_strides = candidate_strides
        self.dilation = dilation
        self.convolution_type = convolution_type
        self.distr_pair = distr_pair

        self.device = device

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
            self.cin,
            (self.input_h, self.input_w),
            self.cout,
            self.convolution_type,
            self.candidate_lengths,
            self.padding_mode,
            self.distr_pair[0].fn,
            self.distr_pair[1].fn,
            self.dilation,
            self.candidate_strides,
            self.device,
            self.random_state
        )

    def forward(self, X):
        """Forward pass through the ROCKET transformation.
        
        Args:
            X (torch.Tensor): Input data of shape [B, C, H, W].
            
        Returns:
            torch.Tensor: Transformed features.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        return self._transform(X.to(self.device))

    def _transform(self, X):
        batch_size, cin, _, _ = X.shape

        output_size = len(self.features_to_extract) * self.cout
        if FeatureType.MAX2D in self.features_to_extract:
            output_size = output_size - self.cout + 4 * self.cout
        if FeatureType.AVG2D in self.features_to_extract:
            output_size = output_size - self.cout + 4 * self.cout

        features = torch.zeros((batch_size, output_size))
        
        s = 0
        for key, var in self.conv_params.items():

            _, dilation_h, dilation_w, padding_h, padding_w, stride = key

            if self.convolution_type == ConvolutionType.SPATIAL:
                padding_1, padding_2 = (0, padding_w), (padding_h, 0)
                dilation_1, dilation_2 = (1, dilation_w), (dilation_h, 1)
                group = 1
            elif self.convolution_type == ConvolutionType.DEPTHWISE_SEP:
                padding_1, padding_2 = (padding_h, padding_w), 0
                dilation_1, dilation_2 = (dilation_h, dilation_w), 1
                group = cin
            elif self.convolution_type == ConvolutionType.DEPTHWISE:
                padding_1 = (padding_h, padding_w)
                dilation_1 = (dilation_h, dilation_w)
                group = cin
            else:
                padding_1 = (padding_h, padding_w)
                dilation_1 = (dilation_h, dilation_w)
                group = 1

            # First convolution
            data = F.conv2d(
                input=X,
                weight=var['weights'][0],
                bias=var['bias'] if self.convolution_type in [
                    ConvolutionType.STANDARD, ConvolutionType.DEPTHWISE] else None,
                stride=stride,
                padding=padding_1,
                dilation=dilation_1,
                groups=group
            )

            # Second convolution if convolution_type requires it
            if self.convolution_type in [ConvolutionType.SPATIAL, ConvolutionType.DEPTHWISE_SEP]:
                data = F.conv2d(
                    input=data,
                    weight=var['weights'][1],
                    bias=var['bias'],
                    stride=stride,
                    padding=padding_2,
                    dilation=dilation_2,
                    groups=1
                )

            # Reconstructing data if depthwise from [B, K*cin, H, W] to [B, K, cin*H*W]
            if self.convolution_type == ConvolutionType.DEPTHWISE:
                kernel_per_channel = data.shape[1] // cin
                merged_kernels = []
                for i in range(kernel_per_channel):
                    merged_kernels.append(
                        data[:, i::kernel_per_channel].flatten(1))
                data = torch.stack(merged_kernels, dim=1)

            features_list = []
            # Computing Max pooling
            if FeatureType.MAX2D in self.features_to_extract:
                max2d = F.adaptive_max_pool2d(data, output_size=2).flatten(1)
                features_list.append(max2d)

            if FeatureType.AVG2D in self.features_to_extract:
                avg2d = F.adaptive_avg_pool2d(data, output_size=2).flatten(1)
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

        return features
