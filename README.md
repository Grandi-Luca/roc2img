# ROCKET for Image Classification (images as volumes)

PyTorch implementation of ROCKET (RandOm Convolutional KErnel Transform) adapted to images by using **3D convolutions**.
For standard 2D images, treat them as volumes with depth $D=1$.

Key points:

- Input tensors are expected in **5D**: `[B, C, D, H, W]`.
- The ROCKET module is a *feature extractor* (no `fit()` / `transform()` methods). Use a downstream classifier (e.g. Ridge).
- Kernel weights/biases can be sampled from multiple distributions, including KDEs fitted from pretrained ResNet weights/biases.

## Installation

Requirements:

- Python **3.10+**
- `torch`, `torchvision`
- `numpy`
- `scikit-learn` (needed both for KDE-based distributions and typical classifiers)

Minimal install:

```bash
pip install numpy torch torchvision scikit-learn
```

Extras used by the provided experiment runner:

```bash
pip install wandb nibabel pandas
```

## Basic Usage

```python
import torch

from rocket_img import ROCKET
from utils import ConvolutionType, DilationType, FeatureType, PaddingMode
from distributions import DistributionType

# Input MUST be 5D: [B, C, D, H, W]
# For standard 2D images use D=1.
B, C, H, W = 100, 3, 32, 32
X_train = torch.rand(B, C, 1, H, W)
y_train = torch.randint(0, 10, (B,))

X_test = torch.rand(20, C, 1, H, W)
y_test = torch.randint(0, 10, (20,))

rocket = ROCKET(
    cin=C,
    input_dhw=(1, H, W),
    cout=1000,
    candidate_lengths=[3],
    candidate_strides=[1],
    padding_mode=PaddingMode.RANDOM,
    distr_pair=(DistributionType.GAUSSIAN_01, DistributionType.UNIFORM),
    features_to_extract=[FeatureType.AVG2D, FeatureType.GMPV, FeatureType.PPV, FeatureType.MPV],
    dilation=DilationType.UNIFORM_ROCKET,
    convolution_type=ConvolutionType.STANDARD,
    random_state=42,
)

with torch.no_grad():
    X_train_features = rocket(X_train).numpy()
    X_test_features = rocket(X_test).numpy()

from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier(alpha=0.1)
clf.fit(X_train_features, y_train)
predictions = clf.predict(X_test_features)
```

Notes:

- Output feature dimension is approximately `cout * (#features)`. Some features (e.g. `AVG2D`, `MAX2D`) produce **4 values per kernel**.
- If you use `DistributionType.REAL_RESNET*_WEIGHT/BIAS`, torchvision may download pretrained weights on first use.

## Main Parameters

### `cin` (int) and `input_dhw` (tuple | int)

- `cin`: number of input channels (e.g. 3 for RGB)
- `input_dhw`: `(D, H, W)` of the input tensor (or a single `int` used for all three)

Inputs must match `[B, cin, D, H, W]`.

### `cout` (int)

Number of random convolutional kernels to generate.

### `candidate_lengths` (list[int])

Kernel sizes to sample.

### `candidate_strides` (list[int])

Stride values to sample.

### `padding_mode` (PaddingMode)

- `ON`: always padded
- `OFF`: never padded
- `RANDOM`: random per-kernel

### `convolution_type` (ConvolutionType)

- `STANDARD`
- `DEPTHWISE`
- `SPATIAL`
- `DEPTHWISE_SEP`

### `features_to_extract` (list[FeatureType])

Common features:

- `PPV`, `MPV`, `MIPV`, `LSPV`
- `GAP`, `MAXPV`, `MINPV`, `GMPV`, `ENTROPY`
- `AVG2D`, `MAX2D` (adaptive pooling to `(1, 2, 2)` → 4 values per kernel)

### `distr_pair` (tuple)

Pair of distributions for (weights, bias). See [distributions.py](distributions.py) for the full list.

Examples:

- Weights: `GAUSSIAN_01`, `CONV2D_WEIGHT`, `REAL_RESNET50_WEIGHT`, `APPROX_RESNET50_WEIGHT_FIRST_GAUSS`, ...
- Bias: `UNIFORM`, `CONV2D_BIAS`, `REAL_RESNET50_BIAS`, `APPROX_RESNET50_BIAS_GLOBAL_GAUSS`, ...

### `dilation` (DilationType)

Supported by the current kernel sampler:

- `UNIFORM_ROCKET` (random dilation like ROCKET)
- `DILATED_1`, `DILATED_2`, `DILATED_3` (fixed dilations)

## Project Structure

```
├── rocket_img.py      # ROCKET feature extractor
├── distributions.py   # Weight/bias sampling distributions
├── utils.py           # Enums + helpers
├── baselines/         # Baseline training code (separate)
└── results/           # Experiment results (CSV)
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- Original ROCKET: [Dempster et al., 2019](https://arxiv.org/abs/1910.13051)