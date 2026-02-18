import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

import time
from typing import Optional

from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score

import argparse
import os
import nibabel as nib

from rocket_img import ROCKET

from distributions import DistributionType
from utils import ConvolutionType, FeatureType, DilationType, _set_random_seed

from ADNI.MultiINPUT_shard_descriptor import MultiINPUTShardDescriptor

import wandb

STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR100 = (0.2675, 0.2565, 0.2761)
MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)

def load_mnist_data(batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.unsqueeze(1)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_cifar10_data(batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    # Preprocess the CIFAR-10 train dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
            transforms.Lambda(lambda x: x.unsqueeze(1)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        './data/cifar10/', train=True, transform=transform, download=True)

    test_set = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False,
                                            download=True, transform=transform)

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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100),
            transforms.Lambda(lambda x: x.unsqueeze(1)),
        ]
    )

    train_set = torchvision.datasets.CIFAR100(
        './data/cifar100/', train=True, transform=transform, download=True)

    test_set = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=False,
                                            download=True, transform=transform)

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

def load_nifti_to_tensor(file_paths):
    """Load multiple NIfTI files and stack them into a single tensor."""
    tensors = []
    for path in file_paths:
        img = nib.load(path)
        data = img.get_fdata()
        tensor = torch.tensor(data, dtype=torch.float32)
        tensors.append(tensor)
    return torch.stack(tensors)

def load_adni_data(
        rank_worldsize:str,
        adni_num:int,
        data_dir:str,
        img_dir:str,
        csv_path:str,
        csv_filename:str,
        data_seed:int,
        batch_size:int=128
    ):
     # Load dataset
    descriptor = MultiINPUTShardDescriptor(
        rank_worldsize= rank_worldsize,
        adni_num= adni_num,
        data_dir= data_dir,
        img_dir= img_dir,
        csv_path= csv_path,
        csv_filename= csv_filename,
        data_seed= data_seed,
    )

    # train e test contengono i path per le immagini da usare come input e le labels associate
    train, test = descriptor.data_by_type['train'], descriptor.data_by_type['val']
    
    base_dir = descriptor.data_dir 
    duplicate_prefix = descriptor.data_dir.split('/')[-1]  # prefisso da rimuovere se presente nei path

    train_paths = [
        os.path.join(base_dir, path.split(f'{duplicate_prefix}/', 1)[-1]) if path.startswith(f'{duplicate_prefix}/') else os.path.join(base_dir, path)
        for path in train['IMG_PATH_NORM_min-max'].tolist()
    ]
    test_paths = [
        os.path.join(base_dir, path.split(f'{duplicate_prefix}/', 1)[-1]) if path.startswith(f'{duplicate_prefix}/') else os.path.join(base_dir, path)
        for path in test['IMG_PATH_NORM_min-max'].tolist()
    ]
    
    # Carica tutte le immagini in un unico tensore
    X_train = load_nifti_to_tensor(train_paths)
    y_train = np.array(train['labels'].tolist())
    
    X_test = load_nifti_to_tensor(test_paths)
    y_test = np.array(test['labels'].tolist())

    train_loader = create_loader(X_train, y_train, batch_size=batch_size)
    test_loader = create_loader(X_test, y_test, batch_size=batch_size)

    return train_loader, test_loader

def create_loader(X, y, batch_size: int):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).long()
    dataset = TensorDataset(X.unsqueeze(1), y)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    return loader

def load_data(dataset: str, batch_size: int, subset_percentage: float = 1.0) -> tuple[DataLoader, DataLoader]:
    if dataset == "mnist":
        return load_mnist_data(batch_size, subset_percentage)
    elif dataset == "cifar10":
        return load_cifar10_data(batch_size, subset_percentage)
    elif dataset == "cifar100":
        return load_cifar100_data(batch_size, subset_percentage)
    elif dataset == "adni":
        return load_adni_data(
            rank_worldsize= '1, 1',
            adni_num = 1,
            data_dir = '/mnt/shared_nfs/brunofolder/MERGE/WALTER/IMGS/a1',
            img_dir = 'ADNI1_ALL_T1',
            csv_path = '/mnt/shared_nfs/brunofolder/MERGE/WALTER/IMGS/ADNI_csv',
            csv_filename = 'ADNI_ready.csv',
            data_seed = 13,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def measure_performance(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader
) -> dict[str, Optional[float]]:
    """Measure the performance of a model on training and testing datasets.
    """

    _, y_train = extract_data_from_loader(train_loader)
    _, y_test = extract_data_from_loader(test_loader)

    # Tempo di trasformazione (feature extraction)
    start = time.time()
    X_train_transformed = []
    for inputs, _ in train_loader:
        X_train_transformed.append(model(inputs.to(model.device)).cpu())
    X_train_transformed = torch.cat(X_train_transformed, dim=0).numpy()
    transform_time_train = time.time() - start

    start = time.time()
    X_test_transformed = []
    for inputs, _ in test_loader:
        X_test_transformed.append(model(inputs.to(model.device)).cpu())
    X_test_transformed = torch.cat(X_test_transformed, dim=0).numpy()
    transform_time_test = time.time() - start

    # Tempo di training ridge regression
    start = time.time()
    clf = RidgeClassifier(alpha=0.1).fit(X_train_transformed, y_train)
    # clf = RidgeClassifierCV(alphas=np.logspace(-3, 0, 100)).fit(X_train_transformed, y_train)
    # clf = LogisticRegression(max_iter=5000).fit(X_train_transformed, y_train)
    training_time = time.time() - start

    # Accuracy ridge regression
    predictions = clf.predict(X_test_transformed)
    acc = accuracy_score(y_test, predictions)

    pred_train = clf.predict(X_train_transformed)
    acc_train = accuracy_score(y_train, pred_train)

    return {
        'transform_time_train': round(transform_time_train, 3),
        'transform_time_test': round(transform_time_test, 3),
        'training_time': round(training_time, 3),
        'solver': 'auto',
        'accuracy': acc,
        'accuracy_train': acc_train,
        'num_params': X_train_transformed.shape[1],
        'classifier': clf.__class__.__name__,
        'alpha': 0.1
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
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "cifar100", "adni"], default="cifar10",
                        help="Dataset to use (default: cifar10)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading")
    parser.add_argument("--subset_percentage", type=float, default=1.0, help="Percentage of dataset to use")
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.dataset, args.batch_size, args.subset_percentage)

    cin, depth, in_height, in_width = train_loader.dataset[0][0].shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')


    f = [
        [FeatureType.AVG2D, FeatureType.PPV, FeatureType.MPV, FeatureType.GMPV],
        # [FeatureType.AVG2D, FeatureType.PPV, FeatureType.MIPV, FeatureType.GMPV],
        [FeatureType.AVG2D, FeatureType.PPV, FeatureType.MPV, FeatureType.MIPV, FeatureType.GMPV],
    ]

    for c in [[5], [7]]:

        for i in f:

            for seed in [0,1,42,7,13]:

                rocket = ROCKET(
                    cin=cin,                          # Number of input channels
                    input_dhw=(depth, in_height, in_width),
                    cout=1000,                          # Number of random kernels
                    candidate_lengths=c,        # Kernel sizes
                    distr_pair=(                        # Distributions for weights and biases
                        DistributionType.GAUSSIAN_01,
                        DistributionType.UNIFORM
                    ),
                    features_to_extract=i,
                    convolution_type=ConvolutionType.STANDARD,
                    dilation=DilationType.UNIFORM_ROCKET,    # Dilation type
                    candidate_strides=[2],
                    device=device,
                    random_state=seed,
                )

                run = wandb.init(
                    project=f"ROCKET_TO_3D_img",
                    entity="luca-gr",
                    config={
                        **rocket.get_params(),
                        'activation_fn': 'relu',
                        'dataset': f'{args.dataset}',
                    },
                )

                metrics = measure_performance(rocket, train_loader, test_loader)

                run.log(metrics)

                run.finish()
