import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import time
from typing import Optional

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

import argparse

from rocket_img import ROCKET

from distributions import DistributionType
from utils import ConvolutionType, FeatureType, DilationType
from utils import _set_random_seed

import wandb

STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR100 = (0.2675, 0.2565, 0.2761)
MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)

def load_mnist_data(batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_cifar10_data(batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    # Preprocess the CIFAR-10 train dataset
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ]
    )

    # Preprocess the CIFAR-10 test dataset
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        './data/cifar10/', train=True, transform=transform_train, download=True)

    test_set = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False,
                                            download=True, transform=transform_test)

    # Trim the training set if required
    if subset_percentage is not None and subset_percentage < 1.0:

        # Compute the number of samples to select
        num_total_samples = len(train_set)
        # Ensure the number of samples does not exceed the total dataset size
        num_samples_to_select = min(
            int(num_total_samples * subset_percentage), num_total_samples)

        _set_random_seed(42)  # For reproducibility
        # Generate random indices to select
        random_indices = torch.randperm(num_total_samples)[
            :num_samples_to_select]

        # Create a subset of the original training set
        reduced_train_set = torch.utils.data.Subset(train_set, random_indices)
        train_set = reduced_train_set

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
        num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        num_workers=2)

    return train_loader, test_loader

def load_cifar100_data(batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    # Preprocess the CIFAR-100 train dataset
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100)]
    )

    # Preprocess the CIFAR-100 test dataset
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100)]
    )

    train_set = torchvision.datasets.CIFAR100(
        './data/cifar100/', train=True, transform=transform_train, download=True)

    test_set = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=False,
                                            download=True, transform=transform_test)

    # Create DataLoader for train and test sets
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    # Trim the training set if required
    if subset_percentage is not None:

        # Compute the number of samples to select
        num_total_samples = len(train_set)
        # Ensure the number of samples does not exceed the total dataset size
        num_samples_to_select = min(
            int(num_total_samples * subset_percentage), num_total_samples)

        _set_random_seed(42)  # For reproducibility
        # Generate random indices to select
        random_indices = torch.randperm(num_total_samples)[
            :num_samples_to_select]

        # Create a subset of the original training set
        reduced_train_set = torch.utils.data.Subset(train_set, random_indices)
        train_set = reduced_train_set

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def load_data(dataset: str, batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    if dataset == "mnist":
        return load_mnist_data(batch_size, subset_percentage)
    elif dataset == "cifar10":
        return load_cifar10_data(batch_size, subset_percentage)
    elif dataset == "cifar100":
        return load_cifar100_data(batch_size, subset_percentage)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def measure_performance(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader
) -> dict[str, Optional[float]]:
    """Measure the performance of a model on training and testing datasets.
    """

    X_train, y_train = extract_data_from_loader(train_loader)
    X_test, y_test = extract_data_from_loader(test_loader)

    start = time.time()
    model.fit(X_train)
    model_fit_time = time.time() - start

    # Tempo di trasformazione (feature extraction)
    start = time.time()
    X_train_transformed = model.transform(train_loader)
    transform_time_train = time.time() - start
    start = time.time()
    X_test_transformed = model.transform(test_loader)
    transform_time_test = time.time() - start

    # Tempo di training ridge regression
    start = time.time()
    clf = RidgeClassifier(alpha=0.1).fit(X_train_transformed, y_train)
    training_time = time.time() - start

    # Accuracy ridge regression
    predictions = clf.predict(X_test_transformed)
    acc = accuracy_score(y_test, predictions)

    return {
        'model_fit_time': round(model_fit_time, 3),
        'transform_time_train': round(transform_time_train, 3),
        'transform_time_test': round(transform_time_test, 3),
        'training_time': round(training_time, 3),
        'solver': 'auto',
        'accuracy': acc,
        'num_params': X_train_transformed.shape[1]
    }


def extract_data_from_loader(data_loader: DataLoader) -> torch.Tensor:
    """Extract data from a DataLoader into a single tensor.

    Args:
        data_loader (DataLoader): The DataLoader to extract data from.

    Returns:
        torch.Tensor: A tensor containing all the data from the DataLoader.
    """
    X_list = []
    y_list = []
    for batch_data, batch_labels in data_loader:
        X_list.append(batch_data)
        y_list.append(batch_labels)
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0).numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ROCKET selected dataset")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "cifar100"], default="cifar10",
                        help="Dataset to use (default: cifar10)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading")
    parser.add_argument("--subset_percentage", type=float, default=1.0, help="Percentage of dataset to use")
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.dataset, args.batch_size, args.subset_percentage)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rocket = ROCKET(
        cout=1000,                          # Number of random kernels
        candidate_lengths=[3],        # Kernel sizes
        distr_pair=(                        # Distributions for weights and biases
            DistributionType.GAUSSIAN_01,
            DistributionType.UNIFORM
        ),
        features_to_extract=[FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV, FeatureType.MIPV, FeatureType.GMPV, FeatureType.LSPV],
        convolution_type=ConvolutionType.STANDARD,
        dilation=DilationType.UNIFORM_ROCKET,    # Dilation type
        candidate_strides=[1],
        device=device
    )

    # f = [[FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV],
    # [FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV, FeatureType.MIPV],
    # [FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV, FeatureType.GMPV],
    # [FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV, FeatureType.MIPV, FeatureType.GMPV],
    # [FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV, FeatureType.LSPV, FeatureType.GMPV],
    # [FeatureType.MAX2D, FeatureType.PPV, FeatureType.MPV, FeatureType.MIPV, FeatureType.GMPV, FeatureType.LSPV]]

    # for i in f:

        # rocket.features_to_extract = i

    for seed in [0, 1, 42]:

        rocket.random_state = seed

        run = wandb.init(
            project=f"rocket2img-{args.dataset}",
            entity="luca-gr",
            config={
                **rocket.get_params(),
            }
        )

        metrics = measure_performance(rocket, train_loader, test_loader)

        run.log(metrics)

        run.finish()

