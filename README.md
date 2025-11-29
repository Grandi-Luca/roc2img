# ROCKET for Image Classification

An implementation of ROCKET (RandOm Convolutional KErnel Transform) for image classification, with support for various weight and bias distributions extracted from pre-trained ResNet models.

## What is ROCKET?

ROCKET is a feature extraction technique based on random convolutions, originally developed for time series. This implementation extends ROCKET to work with 2D images, using convolutional kernels generated from various distributions, including those derived from weights and biases of pre-trained ResNet models.

## Installation

```bash
pip install numpy torch torchvision scikit-learn pandas
```

## Basic Usage

```python
import torch
from rocket_img import ROCKET
from utils import ConvolutionType, FeatureType, DilationType
from distributions import DistributionType

# Create the ROCKET model
rocket = ROCKET(
    cout=1000,                          # Number of random kernels
    candidate_lengths=[3, 5, 7],        # Kernel sizes
    distr_pair=(                        # Distributions for weights and biases
        DistributionType.GAUSSIAN_01,
        DistributionType.UNIFORM
    ),
    features_to_extract=[FeatureType.PPV],  # Features to extract
    dilation=DilationType.UNIFORM_ROCKET,    # Dilation type
    convolution_type=ConvolutionType.STANDARD,
    random_state=42
)

# Sample data
X_train = torch.rand(100, 3, 32, 32)  # [batch, channels, height, width]
y_train = torch.randint(0, 10, (100,))  # 100 labels
X_test = torch.rand(20, 3, 32, 32)    # [batch, channels, height, width]
y_test = torch.randint(0, 10, (20,))    # 20 labels

# Generate kernels (fit)
rocket.fit(X_train)

# Transform data
X_train_features = rocket.transform(X_train)  # Output: numpy array [100, cout]
X_test_features = rocket.transform(X_test)    # Output: numpy array [20, cout]

# Classify with Ridge Regression
from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=0.1)
clf.fit(X_train_features, y_train)
predictions = clf.predict(X_test_features)
```

## Main Parameters

### `cout` (int)
Number of random convolutional kernels to generate. Default: `10000`

### `candidate_lengths` (list)
Candidate sizes for kernels. Default: `[3, 5, 7]`

### `convolution_type` (ConvolutionType)
Convolution type:
- `STANDARD`: Classic 2D convolution
- `DEPTHWISE`: Depthwise convolution (more features, slower)
- `SPATIAL`: Separable kernels (horizontal + vertical)
- `DEPTHWISE_SEP`: Depthwise separable convolution

### `features_to_extract` (list[FeatureType])
Features extracted from each kernel:
- `PPV`: Proportion of Positive Values
- `MPV`: Mean of Positive Values
- `MIPV`: Mean Index of Positive Values
- `LSPV`: Longest Streak of Positive Values

### `distr_pair` (tuple)
Pair of distributions for weights and biases:

**Weight distributions:**
- `GAUSSIAN_01`: Standard Gaussian N(0,1)
- `REAL_RESNET18_WEIGHT`, `REAL_RESNET50_WEIGHT`, `REAL_RESNET101_WEIGHT`, `REAL_RESNET152_WEIGHT`: KDE distributions from real ResNet weights
- `APPROX_RESNET*_WEIGHT_FIRST_GAUSS/LAPLACE`: Parametric approximations

**Bias distributions:**
- `UNIFORM`: Uniform [-1, 1]
- `REAL_RESNET*_BIAS`: KDE distributions from real ResNet biases
- `APPROX_RESNET*_BIAS_GLOBAL_GAUSS/LAPLACE`: Parametric approximations

### `dilation` (DilationType)
Dilation strategy:
- `UNIFORM_ROCKET`: Random dilation (original ROCKET)
- `DILATED_1`, `DILATED_2`, `DILATED_3`: Fixed dilation
- `RANDOM_13`: Random choice among 1, 2, 3

## Project Structure

```
├── rocket_img.py      # Main ROCKET implementation
├── distributions.py   # Distributions for weights and biases
├── utils.py           # Utilities and enumerations
├── test.py            # Test and benchmark script
└── results/           # Experiment results
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- Original ROCKET: [Dempster et al., 2019](https://arxiv.org/abs/1910.13051)
- Image extension based on ResNet **distributions**