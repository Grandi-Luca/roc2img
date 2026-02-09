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

from utils import _set_random_seed

import argparse
import time


STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR100 = (0.2675, 0.2565, 0.2761)
MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)

def load_mnist_data() -> tuple[Dataset, Dataset]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

    test_set = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

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
        './data/cifar10/', train=True, transform=transform_train, download=True)

    test_set = datasets.CIFAR10(root='./data/cifar10/', train=False,
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
        './data/cifar100/', train=True, transform=transform_train, download=True)

    test_set = datasets.CIFAR100(root='./data/cifar100/', train=False,
                                            download=True, transform=transform_test)

    return train_set, test_set


def load_data(dataset: str, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    if dataset == "mnist":
        train_set, test_set = load_mnist_data()
    elif dataset == "cifar10":
        train_set, test_set = load_cifar10_data()
    elif dataset == "cifar100":
        train_set, test_set = load_cifar100_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # obtain training indices that will be used for validation
    _set_random_seed(42)
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