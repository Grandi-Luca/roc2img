# roc2img-1

This repository contains experimental implementations for image feature extraction based on randomized convolutional kernels (ROCKET-style approach) and utilities to train lightweight classifiers on the extracted features. The main component is `RocketNet`, a network that builds fixed-size representations from images using randomized convolutional layers and adaptive pooling.

**Main contents**
- `rocketnet.py`: implementation of `RocketNet`, `RocketLayer` and `SeqRocketLayer` (randomized feature extractors).
- `utils.py`: helpers for custom pooling, random seed control and utility types.
- `baselines/`: scripts and implementations for evaluation and experimental runs.

**Goal**
Provide a fast, optionally deterministic feature extractor that can be paired with linear classifiers (e.g. `RidgeClassifier`) for quick computer vision experiments.

**Minimal installation**
Required packages: Python 3.8+, PyTorch, torchvision, scikit-learn, numpy.

Quick install (no virtualenv shown):

```bash
pip install torch torchvision scikit-learn numpy
```

**Core idea: how `RocketNet` works**

- `RocketLayer` / `SeqRocketLayer`:
  - Initialize convolutional kernels with randomly sampled weights and biases (normal/uniform).
  - Each kernel can have randomized dilation and padding (or deterministic values if specified) to capture responses at multiple scales.
  - After convolution a non-linearity (`ReLU`) is applied and then one or more adaptive pooling operations (e.g. AAP, AMP, AGeM defined in `utils.py`).
  - `SeqRocketLayer` applies convolutions sequentially per output channel; `RocketLayer` groups kernels that share the same dilation/padding parameters to optimize computation and memory.

- `RocketNet`:
  - Composes multiple Rocket blocks/layers (passed as tuples describing class and parameters).
  - Runs in `eval()` mode (used as a feature extractor, not trained via backprop here).
  - Returns the concatenation of features produced by all blocks for each batch example.

**Typical usage**

1. Instantiate `RocketNet` with desired layers and device (CPU/GPU).
2. Feed image batches to obtain fixed feature vectors.
3. Train a simple classifier (e.g. `RidgeClassifier`) on the extracted features.

Minimal example:

```python
import torch
from rocketnet import RocketNet, RocketLayer
from utils import AAP, AMP
from sklearn.linear_model import RidgeClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define two Rocket blocks with adaptive pooling
layers = [
    (RocketLayer, 3, 1000, 7, 1, True, (32, 32), [AAP(1), AMP(1)]),
    (RocketLayer, 3, 2000, 5, 1, True, (32, 32), [AAP(1)])
]

model = RocketNet(device, random_state=42, *layers)

# example batch (B, C, H, W)
x = torch.randn(8, 3, 32, 32).to(device)
features = model(x)  # shape: (B, feature_dim)

# then use an sklearn classifier
clf = RidgeClassifier()
clf.fit(features.cpu().numpy(), y_train)
preds = clf.predict(features_test.cpu().numpy())
```

Note: pooling classes in `utils.py` (`AAP`, `AMP`, `AGeM`, `GeneralizedMeanPooling`) expose `get_output_size()` so that `RocketLayer` can correctly compute the concatenated feature dimension.

**Design choices**
- Convolutions are randomized and not trained: this makes the extractor fast and reliable for quick experiments.
- Layers use random dilation/padding to enrich multi-scale representations.
- The network is evaluated in `eval()` mode and the resulting features can be normalized before feeding them to a classifier.

**Useful files and scripts**
- [rocketnet.py](rocketnet.py): main implementation of `RocketNet`, `RocketLayer`, `SeqRocketLayer`.
- [utils.py](utils.py): custom pooling, random-seed management, and enums.
- [baselines/](baselines/): baseline scripts and experimental runners (see `baselines/exp_run.py`, `baselines/trainer.py`).

**Examples and experiments**
- Example scripts for launching experiments are provided in the `baselines/` folder.
- For reproducible evaluations use `random_state` in `RocketNet` or call `utils.set_random_state(seed)`.

**License**
See the `LICENSE` file for licensing details.

If you want, I can add full run examples, a `requirements.txt` and a step-by-step reproduction guide.
