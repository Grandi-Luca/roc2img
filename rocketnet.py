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
    A layer for the Rocket network that performs convolutional operations with randomized dilation, padding, weights and bias, followed by optional pooling operations. Convolution is performed sequentially for each output channel.
    
    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional operation.
        random_out_dim (bool, optional): Whether to use randomized output dimensions. Defaults to True.
        input_dim (tuple[int, int], optional): Dimensions of the input tensor. Defaults to None.
        poolings (list, optional): List of pooling layers to apply. Defaults to [].
        device (torch.device, optional): Device to run the layer on. Defaults to torch.device('cpu').
    """

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, random_out_dim: bool = True, input_dim: tuple[int, int] = None, poolings: list = [], device: torch.device = torch.device('cpu')):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.random_out_dim = random_out_dim

        assert random_out_dim and len(poolings) > 0, "At least one pooling technique must be specified to standardize the convolutional outputs"
        self.poolings = poolings

        if random_out_dim:
            assert input_dim is not None, "input_dim must be specified if random_out_dim is True"
            max_dilation = (min(list(input_dim)) - 1) / float(self.kernel_size - 1)
            upper = float(np.log2(max_dilation))
            if min(list(input_dim)) > kernel_size:
                self.dilations = torch.from_numpy(np.int32(2 ** np.random.uniform(0.0, upper, size=cout)))
        else:
            self.dilations = torch.ones(cout, dtype=torch.int32)

        candidate_padding = torch.randint(0, 2, (cout,), dtype=torch.bool) if random_out_dim else torch.ones(cout, dtype=torch.bool)
        self.paddings = torch.where(candidate_padding, ((self.kernel_size - 1) * self.dilations) // 2, torch.tensor(0, dtype=torch.int32))

        weights = np.random.normal(0, 1, (cout, cin, self.kernel_size, self.kernel_size)).astype(np.float32)
        weights = weights - weights.mean(axis=(2, 3), keepdims=True)
        bias = np.random.uniform(-1, 1, size=cout).astype(np.float32)

        self.weights = nn.Parameter(torch.from_numpy(weights).to(device))
        self.bias = nn.Parameter(torch.from_numpy(bias).to(device))

    def forward(self, x):
        if len(self.poolings) == 0:
            H_target = math.floor((x.shape[-2] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            W_target = math.floor((x.shape[-1] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            output_size = (self.cout, H_target, W_target)
        else:
            dims = [p.get_output_size() for p in self.poolings]
            flatten = len(set(dims)) > 1 and len(self.poolings) > 1
            output_size = (self.cout * sum(np.prod(dims, axis=-1)),) if flatten else (self.cout * len(self.poolings), *dims[0])

        output = torch.zeros((x.shape[0], *output_size)).to(self.device)

        s = 0
        for i in range(self.cout):
            out = F.conv2d(
                x,
                self.weights[i:i+1],
                self.bias[i:i+1],
                dilation=int(self.dilations[i]),
                padding=int(self.paddings[i]),
                stride=self.stride
            ).relu()

            if len(self.poolings) > 0:
                if flatten:
                    out = torch.cat([pool(out).flatten(1) for pool in self.poolings], dim=1)
                else:
                    out = torch.cat([pool(out) for pool in self.poolings], dim=1)

            output[:, s:s + out.shape[1]] = out
            s += out.shape[1]

        return output

    def extra_repr(self) -> str:
        return (f"cin={self.cin}, cout={self.cout}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"random_out_dim={self.random_out_dim}, "
                f"poolings=[{' '.join([pool.__class__.__name__ for pool in self.poolings])}], "
                f"device={self.device}")

class RocketLayer(nn.Module):
    """A layer for the Rocket network that performs convolutional operations with randomized dilation, padding, weights and bias, followed by optional pooling operations. Kernels are grouped by their dilation and padding parameters.

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
    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, random_out_dim: bool = True, input_dim: tuple[int, int] = None, poolings: list = [], device: torch.device = torch.device('cpu')):
        super().__init__()
        self.cout = cout
        self.cin = cin
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.random_out_dim = random_out_dim

        assert random_out_dim and len(poolings) > 0, "At least one pooling technique must be specified to standardize the convolutional outputs"
        self.poolings = poolings

        if random_out_dim:
            assert input_dim is not None, "input_dim must be specified if random_out_dim is True"
            max_dilation = (min(list(input_dim)) - 1) / float(self.kernel_size - 1)
            upper = float(np.log2(max_dilation))
            if min(list(input_dim)) > kernel_size:
                dilations = np.int32(2 ** np.random.uniform(0.0, upper, size=cout))
        else:
            dilations = np.ones(cout, dtype=np.int32)

        candidate_padding = np.random.randint(2, size=cout, dtype=bool) if random_out_dim else np.ones(cout, dtype=bool)
        paddings = np.where(candidate_padding, ((self.kernel_size - 1) * dilations) // 2, 0)

        key_params = np.column_stack([dilations, paddings])
        unique_params, counts = np.unique(key_params, axis=0, return_counts=True)

        self.convolution_params = {}
        for param, count in zip(unique_params, counts):
            weights = np.random.normal(0, 1, (count, cin, self.kernel_size, self.kernel_size)).astype(np.float32)
            weights = weights - weights.mean(axis=(2, 3), keepdims=True)
            bias = np.random.uniform(-1,1, size=count).astype(np.float32)

            self.convolution_params[tuple(param.tolist())] = (
                torch.from_numpy(weights).to(device),
                torch.from_numpy(bias).to(device),
            )

    def forward(self, x):
        if len(self.poolings) == 0:
            H_target = math.floor((x.shape[-2] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            W_target = math.floor((x.shape[-1] + 2 * ((self.kernel_size - 1)//2) - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
            output_size = (self.cout, H_target, W_target)
        else:
            dims = [p.get_output_size() for p in self.poolings]
            flatten = len(set(dims)) > 1 and len(self.poolings) > 1
            output_size = (self.cout * sum(np.prod(dims, axis=-1)),) if flatten else (self.cout * len(self.poolings), *dims[0])

        output = torch.zeros((x.shape[0], *output_size)).to(self.device)

        s=0
        for (dilation, padding), (weights, bias) in self.convolution_params.items():
            out = F.conv2d(
                x,
                weights,
                bias,
                dilation=dilation,
                padding=padding,
                stride=self.stride
            ).relu()

            if len(self.poolings) > 0:
                if flatten:
                    out = torch.cat([pool(out).flatten(1) for pool in self.poolings], dim=1)
                else:
                    out = torch.cat([pool(out) for pool in self.poolings], dim=1)

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