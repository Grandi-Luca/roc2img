import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import math
import inspect

from utils import set_random_state

class SeqRocketLayer(nn.Module):
    """
    A Rocket layer that applies convolutions with random dilation, padding, weights and bias, then optionally applies pooling.
    Kernels keep their original order. Internally, weights and bias are grouped by the same dilation/padding pair to run the convolutions efficiently.
    After that, the outputs are placed back into the result tensor in the original kernel order.
    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional operation.
        random_out_dim (bool, optional): Whether to use randomized output dimensions. Defaults to True
        input_dim (tuple[int, int, int], optional): Dimensions of the input tensor. Defaults to None.
        poolings (list, optional): List of pooling layers to apply. Defaults to [].
        device (torch.device, optional): Device to run the layer on. Defaults to torch.device('cpu').
    """

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, random_out_dim: bool = True, input_dim: tuple[int, int, int] = None, poolings: list | None = None, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.random_out_dim = random_out_dim

        assert random_out_dim and poolings is not None and len(poolings) > 0, "At least one pooling technique must be specified to standardize the convolutional outputs"
        self.poolings = poolings if poolings.copy() is not None else []

        # Initialize arrays for dilations
        dilations_d = np.ones(cout, dtype=np.int32)
        dilations_hw = np.ones(cout, dtype=np.int32)
        if random_out_dim:
            assert input_dim is not None, "input_dim must be specified if random_out_dim is True"
            input_depth, input_h, input_w = input_dim

            # Compute dilations
            min_input_dim = min(input_depth, input_h, input_w) if input_depth > kernel_size else min(input_h, input_w)
            max_dilation = (min_input_dim - 1) / float(kernel_size - 1) if self.kernel_size > 1 else 1
            upper = float(np.log2(max_dilation)) if max_dilation > 0 else 1
            if min_input_dim > kernel_size:
                d = np.int32(2 ** np.random.uniform(0.0, upper, size=cout))
                dilations_hw = d
                if input_depth > kernel_size:
                    dilations_d = d

        # Compute paddings vectorized
        candidate_pad = np.random.randint(0, 2, size=cout, dtype=bool) if random_out_dim else np.ones(cout, dtype=bool)
        paddings_d = np.where(candidate_pad, ((self.kernel_size - 1) * dilations_d) // 2, 0) if input_depth > self.kernel_size else np.zeros(cout, dtype=np.int32)
        paddings_hw = np.where(candidate_pad, ((self.kernel_size - 1) * dilations_hw) // 2, 0)

        depth = self.kernel_size if input_dim[0] > self.kernel_size else 1
        weights = np.random.normal(0, 1, (cout, cin, depth, self.kernel_size, self.kernel_size)).astype(np.float32)
        weights = weights - weights.mean(axis=(2, 3, 4), keepdims=True)
        bias = np.random.uniform(-1, 1, size=cout).astype(np.float32)

        self.weights = torch.from_numpy(weights).to(device)
        self.bias = torch.from_numpy(bias).to(device)

        # Group kernels by unique parameter combinations
        self.key_params = np.column_stack([dilations_d, dilations_hw, paddings_d, paddings_hw])
        self.unique_params = np.unique(self.key_params, axis=0)

    def forward(self, x):
        flatten = len(self.poolings) > 1
        if len(self.poolings) == 0:
            H_target = math.floor((x.shape[-2] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            W_target = math.floor((x.shape[-1] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            D_target = math.floor((x.shape[-3] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            output_size = (self.cout, D_target, H_target, W_target)
        else:
            dims = [pool.get_output_size() for pool in self.poolings]
            num_features_per_kernel = np.sum([np.prod(d) for d in dims]) # Fixed line
            output_size = (num_features_per_kernel * self.cout,) if flatten else dims[0]

        output = torch.zeros(x.shape[0], *output_size).to(self.device)

        for (dilation_d, dilation_hw, padding_d, padding_hw) in self.unique_params:
            ids = torch.where(torch.all(self.key_params == torch.tensor([dilation_d, dilation_hw, padding_d, padding_hw]), dim=1))[0]

            out = F.conv3d(
                x,
                self.weights[ids],
                self.bias[ids],
                dilation=(dilation_d, dilation_hw, dilation_hw),
                padding=(padding_d, padding_hw, padding_hw),
                stride=self.stride
            ).relu()

            if len(self.poolings) > 0:
                feat_maps = [pool(out).flatten(1) for pool in self.poolings] if flatten else [pool(out) for pool in self.poolings]
                out = torch.cat(feat_maps, dim=-1)

            if flatten:
                offsets = torch.arange(num_features_per_kernel, device=ids.device)
                expanded_ids = (ids.unsqueeze(1) * num_features_per_kernel + offsets).flatten()
                output[:, expanded_ids] = out
            else:
                output[:, ids] = out

        return output

    def extra_repr(self) -> str:
        return (f"cin={self.cin}, cout={self.cout}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"random_out_dim={self.random_out_dim}, "
                f"poolings=[{' '.join([pool.__class__.__name__ for pool in self.poolings])}], "
                f"device={self.device}")

class RocketLayer(nn.Module):
    """A layer for the Rocket network that performs convolutional operations with randomized dilation, padding, weights and bias, followed by optional pooling operations. 
    Kernels are grouped by their dilation and padding parameters.

    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional operation.
        random_out_dim (bool, optional): Whether to use randomized output dimensions. Defaults to True
        input_dim (tuple[int, int], optional): Dimensions of the input tensor. Defaults to None.
        poolings (list, optional): List of pooling layers to apply. Defaults to [].
        device (torch.device, optional): Device to run the layer on. Defaults to torch.device('cpu').
    """
    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, random_out_dim: bool = True, input_dim: tuple[int, int, int] = None, poolings: list = [], device: torch.device = torch.device('cpu')):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.random_out_dim = random_out_dim

        assert random_out_dim and len(poolings) > 0, "At least one pooling technique must be specified to standardize the convolutional outputs"
        self.poolings = poolings

        # Initialize arrays for dilations
        dilations_d = np.ones(cout, dtype=np.int32)
        dilations_hw = np.ones(cout, dtype=np.int32)
        if random_out_dim:
            assert input_dim is not None, "input_dim must be specified if random_out_dim is True"
            input_depth, input_h, input_w = input_dim

            # Compute dilations
            min_input_dim = min(input_depth, input_h, input_w) if input_depth > kernel_size else min(input_h, input_w)
            max_dilation = (min_input_dim - 1) / float(kernel_size - 1) if self.kernel_size > 1 else 1
            upper = float(np.log2(max_dilation)) if max_dilation > 0 else 1
            if min_input_dim > kernel_size:
                d = np.int32(2 ** np.random.uniform(0.0, upper, size=cout))
                dilations_hw = d
                if input_depth > kernel_size:
                    dilations_d = d

        # Compute paddings vectorized
        candidate_pad = np.random.randint(0, 2, size=cout, dtype=bool) if random_out_dim else np.ones(cout, dtype=bool)
        paddings_d = np.where(candidate_pad, ((self.kernel_size - 1) * dilations_d) // 2, 0) if input_depth > self.kernel_size else np.zeros(cout, dtype=np.int32)
        paddings_hw = np.where(candidate_pad, ((self.kernel_size - 1) * dilations_hw) // 2, 0)

        # Group kernels by unique parameter combinations
        key_params = np.column_stack([dilations_d, dilations_hw, paddings_d, paddings_hw])
        unique_params, counts = np.unique(key_params, axis=0, return_counts=True)

        # Generate weights and biases for each unique parameter group
        self.convolution_params = {}
        for key, count in zip(unique_params, counts):
            depth = kernel_size if input_depth > kernel_size else 1
            
            weights = np.random.normal(0, 1, (count, cin, depth, kernel_size, kernel_size)).astype(np.float32)
            weights = weights - weights.mean(axis=(2, 3, 4), keepdims=True)

            bias = np.random.uniform(-1, 1, size=count).astype(np.float32)

            self.convolution_params[tuple(key.tolist())] = (torch.from_numpy(weights).to(device), torch.from_numpy(bias).to(device))

    def forward(self, x):
        flatten = len(self.poolings) > 1
        if len(self.poolings) == 0:
            H_target = math.floor((x.shape[-2] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            W_target = math.floor((x.shape[-1] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            D_target = math.floor((x.shape[-3] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            output_size = (self.cout, D_target, H_target, W_target)
        else:
            dims = [pool.get_output_size() for pool in self.poolings]
            num_features_per_kernel = np.sum([np.prod(d) for d in dims]) # Fixed line
            output_size = (num_features_per_kernel * self.cout,) if flatten else (self.cout, *dims[0])

        output = torch.zeros(x.shape[0], *output_size).to(self.device)

        s=0
        for (dilation_d, dilation_hw, padding_d, padding_hw), (weights, bias) in self.convolution_params.items():
            out = F.conv3d(
                x,
                weights,
                bias,
                dilation=(dilation_d, dilation_hw, dilation_hw),
                padding=(padding_d, padding_hw, padding_hw),
                stride=self.stride
            ).relu()

            if len(self.poolings) > 0:
                feat_maps = [pool(out).flatten(1) for pool in self.poolings] if flatten else [pool(out) for pool in self.poolings]
                out = torch.cat(feat_maps, dim=-1)

            output[:, s:s + out.shape[1]] = out
            s += out.shape[1]
            del out

        return output

    def extra_repr(self) -> str:
        return (f"cin={self.cin}, cout={self.cout}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"random_out_dim={self.random_out_dim}, "
                f"poolings=[{' '.join([pool.__class__.__name__ for pool in self.poolings])}], "
                f"device={self.device}")


def build_module(module: nn.Module, *args, **kwargs):
    return module(*args, **kwargs)


def build_block(*module_list: list, device) -> nn.Sequential:
    def _build(entry):
        if isinstance(entry, nn.Module):
            return entry
        module_cls, *rest = entry if isinstance(entry, tuple) else (entry,)
        kwargs = rest.pop(-1) if rest and isinstance(rest[-1], dict) else {}
        if "device" in inspect.signature(module_cls).parameters:
            kwargs |= {"device": device}
        return build_module(module_cls, *rest, **kwargs)

    return nn.Sequential(*[_build(entry) for entry in module_list])


class RocketNet(nn.Module):
    """The Rocket network architecture, consisting of multiple Rocket layers with specified parameters.
    Args:
        device (torch.device): Device to run the network on.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        *layers (list): Variable number of lists, each containing the parameters for a RocketLayer.
    """
    def __init__(self, device, random_state=None, *layers: list):
        super().__init__()

        self.device = device

        if random_state is not None:
            set_random_state(random_state)

        self.layers = nn.ModuleList([
            build_block(*block, device=device)
            for block in layers
        ])

    def forward(self, x):
        self.eval()
        return torch.cat([block(x) for block in self.layers], dim=1)