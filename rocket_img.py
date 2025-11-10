import numpy as np
import torch
from torch.nn import functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal
import random
import os


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


def _generate_kernels(in_channels, input_size, out_channels, convolution_type, use_bias) -> dict:
    groups = {}

    candidate_sizes = np.array((3, 5, 7), dtype=np.int32)
    # Indices of kernel with padding
    candidate_pad = np.random.randint(2, size=out_channels) == 1

    # Generate random kernel sizes
    sizes = np.random.choice(candidate_sizes, out_channels)

    # Generate dilations and paddings
    dilations = np.zeros(out_channels)
    paddings = np.zeros(out_channels)
    for size in candidate_sizes:
        num_ker = (sizes == size).sum()
        if num_ker > 0:
            # Generate dilations
            dilations[sizes == size] = 2 ** np.random.uniform(
                0, np.log2((input_size - 1) / (size - 1)), size=num_ker)
            dilations[sizes == size] = np.int32(dilations[sizes == size])

            # Generate paddings
            paddings[(sizes == size) & candidate_pad] = (
                (size - 1) * dilations[(sizes == size) & candidate_pad]) // 2

    # Grouping kernel by key: <kernel_size, dilation, padding>
    kernel_params = np.stack(
        [sizes, dilations, paddings], axis=1).astype(np.int32)
    unique_params, counts = np.unique(
        kernel_params, axis=0, return_counts=True)

    for key, c in zip(unique_params, counts):

        # Generate random biases
        bias = None
        if use_bias:
            if convolution_type == "depthwise":
                bias = np.random.uniform(-1, 1, c *
                                         in_channels).astype(np.float32)
            else:
                bias = np.random.uniform(-1, 1, c).astype(np.float32)
            bias = torch.from_numpy(bias)

        groups[tuple(key.tolist())] = {
            "weights": _generate_weights(c, in_channels, key[0], convolution_type),
            "bias": bias,
        }

    return groups


def _generate_weights(out_channels, in_channels, kernel_size, convolution_type=None):

    if convolution_type == "spatial":
        # horizontal kernels
        weights_1 = np.random.normal(
            0, 1, (out_channels, in_channels, 1, kernel_size)).astype(np.float32)
        weights_1 = weights_1 - weights_1.mean(axis=(2, 3), keepdims=True)

        # vertical kernels
        weights_2 = np.random.normal(
            0, 1, (out_channels, out_channels, kernel_size, 1)).astype(np.float32)
        weights_2 = weights_2 - weights_2.mean(axis=(2, 3), keepdims=True)

        return torch.from_numpy(weights_1), torch.from_numpy(weights_2)

    elif convolution_type == "depthwise_sep":
        # depthwise kernels
        weights_1 = np.random.normal(
            0, 1, (in_channels, 1, kernel_size, kernel_size)).astype(np.float32)
        weights_1 = weights_1 - weights_1.mean(axis=(2, 3), keepdims=True)

        # pointwise kernels
        weights_2 = np.random.normal(
            0, 1, (out_channels, in_channels, 1, 1)).astype(np.float32)
        weights_2 = weights_2 - weights_2.mean(axis=(2, 3), keepdims=True)

        return torch.from_numpy(weights_1), torch.from_numpy(weights_2)

    elif convolution_type == "depthwise":
        # depthwise kernels
        weights_1 = np.random.normal(
            0, 1, (out_channels * in_channels, 1, kernel_size, kernel_size)).astype(np.float32)
        weights_1 = weights_1 - weights_1.mean(axis=(2, 3), keepdims=True)

        return torch.from_numpy(weights_1),

    else:
        weights = np.random.normal(
            0, 1, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        weights = weights - weights.mean(axis=(2, 3), keepdims=True)

        return torch.from_numpy(weights),


class ROCKET(BaseEstimator, TransformerMixin):
    def __init__(self, out_channels=10000, convolution_type: Literal['standard', 'spatial', 'depthwise_sep', 'depthwise'] = 'standard', use_bias: bool = True, random_state=None):
        self.out_channels = out_channels
        self.conv_params = {}
        self.convolution_type = convolution_type
        self.random_state = random_state if isinstance(
            random_state, int) else None
        self.use_bias = use_bias

        if self.random_state is not None:
            _set_random_seed(self.random_state)

    def fit(self, X, y=None):
        _, in_channels, _, input_size = X.shape
        self.conv_params = _generate_kernels(
            in_channels, input_size, self.out_channels, self.convolution_type, self.use_bias)

    def transform(self, X):
        batch_size, in_channels, _, input_size = X.shape

        # [B, out_channels]
        features = torch.zeros((batch_size, self.out_channels))

        s = 0
        for (kernel_size, dilation, padding), var in self.conv_params.items():

            output_length = (input_size + (2 * padding)
                             ) - ((kernel_size - 1) * dilation)

            if self.convolution_type == "spatial":
                padding_1, padding_2 = (0, padding), (padding, 0)
                dilation_1, dilation_2 = (1, dilation), (dilation, 1)
                group = 1
            elif self.convolution_type == "depthwise_sep":
                padding_1, padding_2 = (padding, padding), 0
                dilation_1, dilation_2 = (dilation, dilation), 1
                group = in_channels
            elif self.convolution_type == "depthwise":
                padding_1 = (padding, padding)
                dilation_1 = (dilation, dilation)
                group = in_channels
            else:
                padding_1 = (padding, padding)
                dilation_1 = (dilation, dilation)
                group = 1

            # First convolution
            data = F.conv2d(
                input=X,
                weight=var['weights'][0],
                bias=var['bias'] if self.convolution_type in [
                    "standard", "depthwise"] else None,
                stride=1,
                padding=padding_1,
                dilation=dilation_1,
                groups=group
            )

            # Second convolution if convolution_type requires it
            if self.convolution_type in ["spatial", "depthwise_sep"]:
                data = F.conv2d(
                    input=data,
                    weight=var['weights'][1],
                    bias=var['bias'],
                    stride=1,
                    padding=padding_2,
                    dilation=dilation_2,
                    groups=1
                )

            if self.convolution_type == "depthwise":
                kernel_per_layer = var['weights'][0].shape[0] // in_channels
                ppv = torch.zeros(
                    (batch_size, kernel_per_layer))
                for i in range(kernel_per_layer):
                    ppv[:, i] = (data[:, i::kernel_per_layer] > 0).sum(
                        dim=(1, 2, 3)) / output_length

            else:
                # Computing the features: PPV
                ppv = (data > 0).sum(dim=(2, 3)) / \
                    output_length  # [B, num_kernel_group]

            e = s + ppv.shape[1]
            features[:, s:e] = ppv
            s = e

        return features
