from pyexpat import features, model
from sched import scheduler
from unittest import loader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

import argparse
import time


STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR100 = (0.2675, 0.2565, 0.2761)
MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)

def load_mnist_data() -> tuple[Dataset, Dataset]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(root='../data/', train=True, download=True, transform=transform)

    test_set = datasets.MNIST(root='../data/', train=False, download=True, transform=transform)

    return train_set, test_set

def load_cifar10_data() -> tuple[Dataset, Dataset]:
    # Preprocess the CIFAR-10 train dataset
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)]
    )

    # Preprocess the CIFAR-10 test dataset
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)]
    )

    train_set = datasets.CIFAR10(
        '../data/cifar10/', train=True, transform=transform_train, download=True)

    test_set = datasets.CIFAR10(root='../data/cifar10/', train=False,
                                            download=True, transform=transform_test)

    return train_set, test_set

def load_cifar100_data() -> tuple[Dataset, Dataset]:
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

    train_set = datasets.CIFAR100(
        '../data/cifar100/', train=True, transform=transform_train, download=True)

    test_set = datasets.CIFAR100(root='../data/cifar100/', train=False,
                                            download=True, transform=transform_test)

    return train_set, test_set

def load_adni_data(
        rank_worldsize:str,
        adni_num:int,
        data_dir:str,
        img_dir:str,
        csv_path:str,
        csv_filename:str,
        data_seed:int,
        batch_size:int=4
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

def load_data(dataset: str, batch_size: int=128) -> tuple[DataLoader, DataLoader, DataLoader]:
    if dataset == "mnist":
        train_set, test_set = load_mnist_data()
    elif dataset == "cifar10":
        train_set, test_set = load_cifar10_data()
    elif dataset == "cifar100":
        train_set, test_set = load_cifar100_data()
    elif dataset == "adni":
        train_set, test_set = load_adni_data(
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
    
    # obtain training indices that will be used for validation
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=2)
    valid_loader = DataLoader(train_set, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
        num_workers=2)
    
    return train_loader, valid_loader, test_loader