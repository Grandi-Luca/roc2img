import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision

import torchvision.transforms as transforms

import time

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from typing import Optional

from rocket_img import ROCKET

from distributions import DistributionType, extract_weights_and_biases, fit_kde_models
from utils import _set_random_seed
from utils import ConvolutionType, FeatureType, PaddingMode, DilationType, ResNetModel


def load_data(batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    """ Load CIFAR-10 dataset and create DataLoaders for training and testing.

    Args:
        batch_size (int): The number of samples per batch.
        subset_percentage (float, optional): The percentage of the training set to use. Defaults to 1.0.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and testing DataLoaders.
    """
    # Preprocess the CIFAR-10 train dataset
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    # Preprocess the CIFAR-10 test dataset
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    train_set = torchvision.datasets.CIFAR10(
        './data/cifar10/', train=True, transform=transform_train, download=True)

    test_set = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False,
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


def measure_performance(
    model,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> dict[str, Optional[float]]:
    """Measure the performance of a model on training and testing datasets.

    Args:
        model (object): The model to be evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        y_train (np.ndarray): Labels for the training dataset.
        y_test (np.ndarray): Labels for the testing dataset.
    Returns:
        dict[str, Optional[float]]: A dictionary containing performance metrics.
    """

    start = time.time()
    model.fit(torch.rand(1, 3, 32, 32))  # Dummy fit to initialize parameters
    model_fit_time = time.time() - start

    # Tempo di trasformazione (feature extraction)
    start = time.time()
    X_train_transformed = model.transform(X_train)
    transform_time_train = time.time() - start
    start = time.time()
    X_test_transformed = model.transform(X_test)
    transform_time_test = time.time() - start

    # Tempo di training ridge regression
    start = time.time()
    ridge = RidgeClassifier(alpha=0.1).fit(X_train_transformed, y_train)
    training_time_ridge = time.time() - start

    # preds = ridge.predict(X_train_transformed)
    # acc_train_ridge = accuracy_score(y_train, preds)

    # Accuracy ridge regression
    predictions_ridge = ridge.predict(X_test_transformed)
    acc_ridge = accuracy_score(y_test, predictions_ridge)

    return {
        'model_fit_time': round(model_fit_time, 3),
        'transform_time_train': round(transform_time_train, 3),
        'transform_time_test': round(transform_time_test, 3),
        'training_time_ridge': round(training_time_ridge, 3),
        'accuracy_ridge': acc_ridge,
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

    batch_size = 128
    subset_percentage = 0.5

    all_results = []  # raccoglie tutte le esecuzioni

    train_loader, test_loader = load_data(batch_size, subset_percentage)
    # Extract labels for the entire training and test sets
    X_train, y_train = extract_data_from_loader(train_loader)
    X_test, y_test = extract_data_from_loader(test_loader)


    rocket = ROCKET(
        cout=1000,
        candidate_lengths=[3,5],
        distr_pair=(DistributionType.REAL_RESNET101_WEIGHT, DistributionType.REAL_RESNET101_BIAS),
        features_to_extract=[FeatureType.PPV],
        dilation=DilationType.UNIFORM_ROCKET,
    )

    for convolution_type in [ConvolutionType.STANDARD, ConvolutionType.DEPTHWISE]:

        for seed in [0, 1, 42]:

            rocket.random_state = seed
            rocket.convolution_type = convolution_type

            metrics = measure_performance(
                rocket, X_train, X_test, y_train, y_test)
            all_results.append({
                'subset_percentage': subset_percentage,
                **rocket.get_params(),
                **metrics,
            })

            # fix feature_to_extract representation for DataFrame
            all_results[-1]['features_to_extract'] = " - ".join(
                all_results[-1]['features_to_extract'])
            # fix distr_pair representation for DataFrame
            all_results[-1]['distr_pair'] = " - ".join(
                [str(dp.name) for dp in all_results[-1]['distr_pair']])
            all_results[-1]['weight_distribution'] = all_results[-1]['distr_pair'].split(" - ")[0]
            all_results[-1]['bias_distribution'] = all_results[-1]['distr_pair'].split(" - ")[1]
            all_results[-1]['candidate_lengths'] = " - ".join(
                [str(cl) for cl in all_results[-1]['candidate_lengths']])

    results_df = pd.DataFrame(all_results)

    metric_cols = ['model_fit_time', 'transform_time_train', 'transform_time_test',
                   'training_time_ridge', 'accuracy_ridge']

    grouped_results = (
        results_df
        .groupby(['subset_percentage', 'convolution_type', 'candidate_lengths', 'features_to_extract', 'distr_pair', 'dilation', 'cout'])[metric_cols]
        .agg(['mean', 'std'])
        .round(3)
    )

    grouped_results.index.set_names(
        ['subset_percentage', 'convolution_type', 'candidate_lengths', 'features_to_extract', 'distr_pair', 'dilation', 'cout'], inplace=True)

    grouped_results.to_csv(
        './results/test_1.csv',
        float_format='%.3f'
    )
