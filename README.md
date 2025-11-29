# ROCKET for Image Classification

Un'implementazione di ROCKET (RandOm Convolutional KErnel Transform) per la classificazione di immagini, con supporto per diverse distribuzioni di pesi e bias estratte da modelli ResNet pre-addestrati.

## Cos'è ROCKET?

ROCKET è una tecnica di estrazione di caratteristiche basata su convoluzioni casuali, originariamente sviluppata per serie temporali. Questa implementazione estende ROCKET per lavorare con immagini 2D, utilizzando kernel convoluzionali generati da varie distribuzioni, incluse quelle derivate dai pesi e bias di modelli ResNet pre-addestrati.

## Installazione

```bash
pip install numpy torch torchvision scikit-learn pandas
```

## Utilizzo Base

```python
import torch
from rocket_img import ROCKET
from utils import ConvolutionType, FeatureType, DilationType
from distributions import DistributionType

# Crea il modello ROCKET
rocket = ROCKET(
    cout=1000,                          # Numero di kernel casuali
    candidate_lengths=[3, 5, 7],        # Dimensioni dei kernel
    distr_pair=(                        # Distribuzioni per pesi e bias
        DistributionType.GAUSSIAN_01,
        DistributionType.UNIFORM
    ),
    features_to_extract=[FeatureType.PPV],  # Features da estrarre
    dilation=DilationType.UNIFORM_ROCKET,    # Tipo di dilation
    convolution_type=ConvolutionType.STANDARD,
    random_state=42
)

# Genera i kernel (fit)
X_train = torch.rand(100, 3, 32, 32)  # [batch, channels, height, width]
rocket.fit(X_train)

# Trasforma i dati
X_train_features = rocket.transform(X_train)  # Output: numpy array [100, cout]

# Classifica con Ridge Regression
from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=0.1)
clf.fit(X_train_features, y_train)
predictions = clf.predict(X_test_features)
```

## Parametri Principali

### `cout` (int)
Numero di kernel convoluzionali casuali da generare. Default: `10000`

### `candidate_lengths` (list)
Dimensioni candidate per i kernel. Default: `[3, 5, 7]`

### `convolution_type` (ConvolutionType)
Tipo di convoluzione:
- `STANDARD`: Convoluzione 2D classica
- `DEPTHWISE`: Convoluzione depthwise (più features, più lento)
- `SPATIAL`: Kernel separabili (orizzontale + verticale)
- `DEPTHWISE_SEP`: Depthwise separable convolution

### `features_to_extract` (list[FeatureType])
Features estratte da ogni kernel:
- `PPV`: Proportion of Positive Values (proporzione di valori positivi)
- `MPV`: Mean of Positive Values (media dei valori positivi)
- `MIPV`: Mean Index of Positive Values
- `LSPV`: Longest Streak of Positive Values

### `distr_pair` (tuple)
Coppia di distribuzioni per pesi e bias:

**Distribuzioni per i pesi:**
- `GAUSSIAN_01`: Gaussiana standard N(0,1)
- `REAL_RESNET18_WEIGHT`, `REAL_RESNET50_WEIGHT`, `REAL_RESNET101_WEIGHT`, `REAL_RESNET152_WEIGHT`: Distribuzioni KDE dai pesi reali di ResNet
- `APPROX_RESNET*_WEIGHT_FIRST_GAUSS/LAPLACE`: Approssimazioni parametriche

**Distribuzioni per i bias:**
- `UNIFORM`: Uniforme [-1, 1]
- `REAL_RESNET*_BIAS`: Distribuzioni KDE dai bias reali di ResNet
- `APPROX_RESNET*_BIAS_GLOBAL_GAUSS/LAPLACE`: Approssimazioni parametriche

### `dilation` (DilationType)
Strategia di dilation:
- `UNIFORM_ROCKET`: Dilation casuale (originale ROCKET)
- `DILATED_1`, `DILATED_2`, `DILATED_3`: Dilation fissa
- `RANDOM_13`: Scelta casuale tra 1, 2, 3

## Struttura del Progetto

```
├── rocket_img.py      # Implementazione principale di ROCKET
├── distributions.py   # Distribuzioni per pesi e bias
├── utils.py           # Utility e enumerazioni
├── test.py            # Script di test e benchmark
└── results/           # Risultati degli esperimenti
```

## Licenza

Apache License 2.0 - vedi [LICENSE](LICENSE) per dettagli.

## Riferimenti

- ROCKET originale: [Dempster et al., 2019](https://arxiv.org/abs/1910.13051)
- Estensione per immagini basata su distribuzioni ResNet