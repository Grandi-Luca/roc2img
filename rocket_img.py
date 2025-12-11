import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.base import BaseEstimator, TransformerMixin

from typing import List

from utils import ConvolutionType, FeatureType
from utils import _set_random_seed, ResNetModel
from distributions import DistributionType, extract_weights_and_biases, fit_kde_models


def _fit_dilations(input_length, num_kernels, num_features, kernel_size, max_dilations_per_kernel):

    assert num_kernels <= num_features, "Number of kernels must be less than or equal to number of features."
    assert kernel_size > 1, "Kernel size must be greater than 1."

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (kernel_size - 1))

    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (
        num_features_per_dilation * multiplier).astype(np.int32)

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _extract_features(data: torch.Tensor, features_to_extract: List[FeatureType], output_length: int) -> torch.Tensor:
    assert data.ndim == 3, "Data must be a 3D tensor [B, K, N]"

    # Positive indices
    pos_mask = data > 0
    pos_count = pos_mask.sum(-1)

    if FeatureType.MPV in features_to_extract or FeatureType.MIPV in features_to_extract:
        zero_mask = pos_count == 0
        pos_count_clone = pos_count.to(torch.float32)
        pos_count_clone[zero_mask] = 1  # to avoid division by zero

    # Extract features
    features_list = []
    for feature in features_to_extract:

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
            gmpv = ((data**p).mean(-1)) ** (1/p)
            features_list.append(gmpv)

        if feature == FeatureType.ENTROPY:
            eps = 1e-12
            p = torch.softmax(data, dim=-1)
            entropy = - (p * (p + eps).log()).sum(dim=-1)
            features_list.append(entropy)

    return torch.cat(features_list, dim=1)


class ROCKET(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cout=100,
        kernel_size=3,
        distr_pair: tuple[DistributionType, DistributionType] = (
            DistributionType.GAUSSIAN_01, DistributionType.UNIFORM),
        stride: int = 1,
        num_features: int = 10000,
        max_dilations_per_kernel: int = 32,
        features_to_extract: list[FeatureType] = [FeatureType.PPV],
        random_state=None
    ):

        self.cout = cout
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_features = num_features
        self.convolution_type = ConvolutionType.STANDARD
        self.distr_pair = distr_pair
        self.max_dilations_per_kernel = max_dilations_per_kernel

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

    def fit(self, X, y=None):
        """Generate the random convolutional kernels.

        Args:
            X (torch.Tensor): Input data of shape [B, num_kernels, H, W].
            y (torch.Tensor, optional): Parameter used just for compatibility. Defaults to None.
        """
        _, cin, _, input_size = X.shape
        if self.random_state is not None:
            _set_random_seed(self.random_state)

        dilations, num_features_per_dilation = _fit_dilations(
            input_size, self.cout, self.num_features, self.kernel_size, self.max_dilations_per_kernel)

        # Generate random bias
        bias = self.distr_pair[1].fn(
            self.cout * sum(num_features_per_dilation), cin, self.kernel_size, self.kernel_size)

        # Generate random weights
        weights = self.distr_pair[0].fn(
            self.cout, cin, self.kernel_size, self.kernel_size)

        self.dilations = dilations
        self.num_features_per_dilation = num_features_per_dilation
        self.bias = bias
        self.weights = weights

    def transform(self, X):
        new_batch_size = 128

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        y = torch.zeros(X.shape[0], dtype=torch.int32)  # dummy labels
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset, batch_size=new_batch_size, shuffle=False, num_workers=2)

        X_list = []

        for X_batch, _ in loader:
            X_batch_transformed = self._transform(X_batch)
            X_list.append(X_batch_transformed.cpu())
        return torch.cat(X_list, dim=0).numpy()

    def _transform(self, X):
        batch_size, _, _, _ = X.shape

        total_features = np.sum(
            self.num_features_per_dilation) * self.cout * len(self.features_to_extract)

        # [B, total_features]
        features = torch.zeros((batch_size, total_features))

        s = 0
        for dil_idx, dilation in enumerate(self.dilations):

            num_features_cur_dilation = self.num_features_per_dilation[dil_idx]
            num_features_prev_dilation = self.num_features_per_dilation[dil_idx -
                                                                        1] if dil_idx > 0 else 0

            _padding0 = (dil_idx % 2) == 0

            # Helper function to process convolution
            def _process_conv(X, padding, num_repeats, bias_start_idx, bias_end_idx, additional_element=False):
                padding = ((self.kernel_size - 1)
                           * dilation) // 2 if padding else 0

                data = F.conv2d(
                    input=X,
                    weight=self.weights,
                    padding=padding,
                    dilation=dilation,
                    stride=self.stride,
                    groups=1
                )

                output_length = data.shape[-1]

                if num_repeats > 1:
                    data = data.repeat(1, num_repeats, 1, 1)

                if additional_element:
                    data = torch.cat((data, data[:, :1, :, :]), dim=1)

                data = data.flatten(2)
                data = data + \
                    self.bias[bias_start_idx:bias_end_idx].view(1, -1, 1)

                return _extract_features(data, self.features_to_extract, output_length)

            start_bias_idx = num_features_prev_dilation * self.cout
            mid_bias_idx = start_bias_idx + (num_features_cur_dilation // 2) * \
                self.cout + (num_features_cur_dilation % 2)

            extracted_features = _process_conv(
                X,
                _padding0,
                num_features_cur_dilation // 2,
                start_bias_idx,
                mid_bias_idx,
                (num_features_cur_dilation % 2 == 1)
            )

            if num_features_cur_dilation > 1:
                end_bias_idx = mid_bias_idx + self.cout * \
                    (num_features_cur_dilation // 2)
                extracted_features_2 = _process_conv(
                    X, not _padding0, num_features_cur_dilation // 2, mid_bias_idx, end_bias_idx
                )
                extracted_features = torch.cat(
                    (extracted_features, extracted_features_2), dim=1
                )

            e = s + extracted_features.shape[1]
            features[:, s:e] = extracted_features
            s = e

        return features
