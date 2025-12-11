import numpy as np

import torch
import random
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.base import BaseEstimator, TransformerMixin

from collections.abc import Callable

from typing import Optional, List

from utils import ConvolutionType, PaddingMode, FeatureType, DilationType
from utils import _set_random_seed, ResNetModel
from distributions import DistributionType, extract_weights_and_biases, fit_kde_models


def _generate_kernels(
    cin: int,
    input_size: int,
    cout: int,
    convolution_type: ConvolutionType,
    candidate_sizes: list,
    padding_mode: PaddingMode,
    weight_distr_fn: Callable,
    bias_distr_fn: Callable,
    dilation: DilationType,
    random_state: Optional[int] = None
) -> dict:

    if random_state is not None:
        _set_random_seed(random_state)

    groups = {}

    # Indices of kernel with padding
    if padding_mode == PaddingMode.ON:
        candidate_pad = np.ones(cout, dtype=bool)
    elif padding_mode == PaddingMode.OFF:
        candidate_pad = np.zeros(cout, dtype=bool)
    else:  # RANDOM
        candidate_pad = np.random.randint(2, size=cout) == 1

    # Generate random kernel sizes based on candidate sizes (equal probability)
    sizes = np.random.choice(candidate_sizes, cout)

    # Generate dilations and paddings
    dilations = np.zeros(cout)
    paddings = np.zeros(cout)
    for size in candidate_sizes:
        num_ker = (sizes == size).sum()
        if num_ker > 0:
            # Generate dilations
            if size == 1:
                dilations[sizes == size] = 1
            elif isinstance(dilation, int):
                dilations[sizes == size] = dilation
            else:
                if dilation == DilationType.RANDOM_13:
                    dilations[sizes == size] = np.random.choice([1,2,3], size=num_ker)
                else:
                    # Rocket dilation
                    dilations[sizes == size] = 2 ** np.random.uniform(
                        0, np.log2((input_size - 1) / (size - 1)), size=num_ker)
            dilations[sizes == size] = np.int32(dilations[sizes == size])

            # Compute paddings for kernels with padding
            paddings[(sizes == size) & candidate_pad] = (
                (size - 1) * dilations[(sizes == size) & candidate_pad]) // 2

    # Grouping kernel by key: <kernel_size, dilation, padding>
    kernel_params = np.stack(
        [sizes, dilations, paddings], axis=1).astype(np.int32)
    unique_params, counts = np.unique(
        kernel_params, axis=0, return_counts=True)

    for key, c in zip(unique_params, counts):

        # Generate random biases
        if convolution_type == ConvolutionType.DEPTHWISE:
            bias = bias_distr_fn(
                c * cin, cin, key[0], key[0], groups=cin)
        else:
            bias = bias_distr_fn(c, cin, key[0], key[0])

        # Generate weights
        if convolution_type == ConvolutionType.SPATIAL:
            # horizontal kernels
            weights_1 = weight_distr_fn(c, cin, 1, key[0])
            # vertical kernels
            weights_2 = weight_distr_fn(c, cin, key[0], 1)
            weights = (weights_1, weights_2)

        elif convolution_type == ConvolutionType.DEPTHWISE_SEP:
            # depthwise kernels
            weights_1 = weight_distr_fn(
                cin, cin, key[0], key[0], groups=cin)
            # pointwise kernels
            weights_2 = weight_distr_fn(c, cin, 1, 1)
            weights = (weights_1, weights_2)

        elif convolution_type == ConvolutionType.DEPTHWISE:
            # depthwise kernels
            weights_1 = weight_distr_fn(
                c * cin, cin, key[0], key[0], groups=cin)
            weights = (weights_1,)

        else:
            weights_1 = weight_distr_fn(c, cin, key[0], key[0])
            weights = (weights_1,)

        groups[tuple(key.tolist())] = {
            "weights": weights,
            "bias": bias,
        }

    return groups

class ROCKET(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cout=10000,
            candidate_lengths: list[int] = [3, 5, 7],
            padding_mode: PaddingMode = PaddingMode.RANDOM,
            distr_pair: tuple[DistributionType, DistributionType] = (DistributionType.GAUSSIAN_01, DistributionType.UNIFORM),
            dilation: DilationType = DilationType.UNIFORM_ROCKET,
            stride: int = 1,
            convolution_type: ConvolutionType = ConvolutionType.STANDARD,
            features_to_extract: list[FeatureType] = [FeatureType.PPV],
            random_state=None):

        self.cout = cout
        self.candidate_lengths = candidate_lengths
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.convolution_type = convolution_type
        self.distr_pair = distr_pair

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

        self.conv_params = {}

    def fit(self, X, y=None):
        """Generate the random convolutional kernels.

        Args:
            X (torch.Tensor): Input data of shape [B, C, H, W].
            y (torch.Tensor, optional): Parameter used just for compatibility. Defaults to None.
        """
        _, cin, _, input_size = X.shape
        self.conv_params = _generate_kernels(
            cin,
            input_size,
            self.cout,
            self.convolution_type,
            self.candidate_lengths,
            self.padding_mode,
            self.distr_pair[0].fn,
            self.distr_pair[1].fn,
            self.dilation,
            self.random_state
        )

    def transform(self, X):
        new_batch_size = 128
        max_batch_size = 2**13

        batch_size, _, _, _ = X.shape
        if (np.log2(batch_size) % 1 != 0 or batch_size > max_batch_size):
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            y = torch.zeros(X.shape[0], dtype=torch.int32)  # dummy labels
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=new_batch_size, shuffle=False, num_workers=2)

            X_list = []

            for X_batch, _ in loader:
                X_batch_transformed = self._transform(X_batch)
                X_list.append(X_batch_transformed.cpu())
            return torch.cat(X_list, dim=0).numpy()

        else:
            return self._transform(X).cpu().numpy()


    def _transform(self, X):
        batch_size, cin, _, input_size = X.shape

        # [B, cout]
        features = torch.zeros(
            (batch_size, len(self.features_to_extract) * self.cout))

        s = 0
        for (kernel_size, dilation, padding), var in self.conv_params.items():

            output_length = ((input_size + (2 * padding)
                              ) - ((kernel_size - 1) * dilation) - 1) // self.stride + 1

            if self.convolution_type == ConvolutionType.SPATIAL:
                padding_1, padding_2 = (0, padding), (padding, 0)
                dilation_1, dilation_2 = (1, dilation), (dilation, 1)
                group = 1
            elif self.convolution_type == ConvolutionType.DEPTHWISE_SEP:
                padding_1, padding_2 = (padding, padding), 0
                dilation_1, dilation_2 = (dilation, dilation), 1
                group = cin
            elif self.convolution_type == ConvolutionType.DEPTHWISE:
                padding_1 = (padding, padding)
                dilation_1 = (dilation, dilation)
                group = cin
            else:
                padding_1 = (padding, padding)
                dilation_1 = (dilation, dilation)
                group = 1

            # First convolution
            data = F.conv2d(
                input=X,
                weight=var['weights'][0],
                bias=var['bias'] if self.convolution_type in [
                    ConvolutionType.STANDARD, ConvolutionType.DEPTHWISE] else None,
                stride=self.stride,
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
                    stride=self.stride,
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

            # [B, num_kernel_group, H*W]
            data = data.flatten(2)
            # Positive indices
            pos_mask = data > 0
            pos_count = pos_mask.sum(-1)


            if FeatureType.MPV in self.features_to_extract or FeatureType.MIPV in self.features_to_extract:
                zero_mask = pos_count == 0
                pos_count_clone = pos_count.to(torch.float32)
                pos_count_clone[zero_mask] = 1  # to avoid division by zero

            # Extract features
            features_list = []
            for feature in self.features_to_extract:

                # Computing PPV
                if feature == FeatureType.PPV:
                    ppv = pos_count.to(torch.float32) / float(output_length)
                    features_list.append(ppv)

                # Computing MPV
                if feature == FeatureType.MPV:
                    mpv_num = (data*pos_mask).sum(-1)
                    mpv = mpv_num / pos_count_clone
                    # If no positive, set to 0
                    mpv[zero_mask] = 0
                    features_list.append(mpv)

                # Computing MIPV
                if feature == FeatureType.MIPV:
                    N = data.shape[-1]
                    idx = torch.arange(N, dtype=torch.float32)
                    mipv_num = pos_mask.to(torch.float32) @ idx   # [B, K]
                    mipv = mipv_num / pos_count_clone
                    mipv[zero_mask] = -1.0
                    features_list.append(mipv)

                # Computing LSPV
                if feature == FeatureType.LSPV:
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

                # Computing Max Pooling
                if feature == FeatureType.MAXPV:
                    max_val = data.max(-1)
                    maxpv = max_val.values
                    features_list.append(maxpv)

                # Computing Min Pooling
                if feature == FeatureType.MINPV:
                    min_val = data.min(-1)
                    minpv = min_val.values
                    features_list.append(minpv)

                # Computing Generalized Mean Pooling
                if feature == FeatureType.GMPV:
                    p = 2.0
                    gmpv = ( (data**p).mean(-1) ) **(1/p)
                    features_list.append(gmpv)    

                if feature == FeatureType.ENTROPY:
                    eps = 1e-12
                    p = torch.softmax(data, dim=-1)
                    entropy = - (p * (p + eps).log()).sum(dim=-1)
                    features_list.append(entropy)

            extracted_features = torch.cat(features_list, dim=1)
            e = s + extracted_features.shape[1]
            features[:, s:e] = extracted_features
            s = e

        return features
