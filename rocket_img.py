import numpy as np

import torch
from torch.nn import functional as F

from sklearn.base import BaseEstimator, TransformerMixin

from typing import Optional
from collections.abc import Callable

from utils import ConvolutionType, PaddingMode, FeatureType, DilationType
from utils import _set_random_seed
from distributions import DistributionType


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
) -> dict:
    """ Generate random convolutional kernels for ROCKET.

    Args:
        cin (int): input channels
        input_size (int): input size (height/width)
        cout (int): number of kernels to generate
        convolution_type (ConvolutionType): type of convolution
        candidate_sizes (np.ndarray): candidate kernel sizes
        padding_mode (PaggingMode): padding mode

    Returns:
        dict: Dictionary containing generated kernels and biases.
    """

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
            if isinstance(dilation, int):
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

        groups[tuple(key.tolist())] = {
            "weights": _generate_weights(c, cin, key[0], weight_distr_fn, convolution_type),
            "bias": bias,
        }

    return groups


def _generate_weights(
    cout: int,
    cin: int,
    kernel_size: int,
    weight_distr_fn: Callable,
    convolution_type: ConvolutionType = ConvolutionType.STANDARD
):
    if convolution_type == ConvolutionType.SPATIAL:
        # horizontal kernels
        weights_1 = weight_distr_fn(
            cout, cin, 1, kernel_size)
        # vertical kernels
        weights_2 = weight_distr_fn(
            cout, cin, kernel_size, 1)
        return weights_1, weights_2

    elif convolution_type == ConvolutionType.DEPTHWISE_SEP:
        # depthwise kernels
        weights_1 = weight_distr_fn(
            cin, cin, kernel_size, kernel_size, groups=cin)
        # pointwise kernels
        weights_2 = weight_distr_fn(cout, cin, 1, 1)
        return weights_1, weights_2

    elif convolution_type == ConvolutionType.DEPTHWISE:
        # depthwise kernels
        weights_1 = weight_distr_fn(
            cout * cin, cin, kernel_size, kernel_size, groups=cin)
        return weights_1,

    else:
        weights = weight_distr_fn(
            cout, cin, kernel_size, kernel_size)
        return weights,


class ROCKET(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            distr_pair: tuple[DistributionType, DistributionType],
            cout=10000,
            candidate_lengths: list = [3, 5, 7],
            padding_mode: PaddingMode = PaddingMode.RANDOM,
            dilation: DilationType = DilationType.UNIFORM_ROCKET,
            stride: int = 1,
            convolution_type: ConvolutionType = ConvolutionType.STANDARD,
            features_to_extract: list[FeatureType] = [FeatureType.PPV],
            random_state=None):
        """ Initialize the ROCKET transformer.

        Args:
            weight_distribution_fn (Callable): Function to sample weights.
            bias_distribution_fn (Callable): Function to sample biases.
            cout (int, optional): Number of convolutional kernels. Defaults to 10000.
            candidate_lengths (np.ndarray, optional): Candidate kernel sizes. Defaults to np.array((3,5,7)).
            padding_mode (PaddingMode, optional): Padding mode. Defaults to PaddingMode.RANDOM.
            dilation (Optional[int], optional): Dilation for the kernels. Defaults to None.
            stride (int, optional): Stride for the convolutions. Defaults to 1.
            convolution_type (ConvolutionType, optional): Type of convolution. Defaults to ConvolutionType.STANDARD.
            feature_to_extract (list[FeatureType], optional): Features to extract. Defaults to [FeatureType.PPV].
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        """

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
        if self.random_state is not None:
            _set_random_seed(self.random_state)

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
            self.dilation
        )

    def transform(self, X):
        """Transform the input data using the generated convolutional kernels.
        Args:
            X (torch.Tensor): Input data of shape [B, C, H, W].
        Returns:
            torch.Tensor: Transformed data of shape [B, cout].
        """
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
                stride=1,
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
                    stride=1,
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
                    # If no positive, set to -1
                    mpv[zero_mask] = -1.0
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

            extracted_features = torch.cat(features_list, dim=1)
            e = s + extracted_features.shape[1]
            features[:, s:e] = extracted_features
            s = e

        return features
