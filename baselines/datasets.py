import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
import os
from utils import MultiINPUTShardDescriptor
import nibabel as nib
from torchvision import datasets, transforms


def load_nifti_to_tensor(file_paths):
    """Load multiple NIfTI files and stack them into a single tensor."""
    tensors = []
    for path in file_paths:
        img = nib.load(path)
        data = img.get_fdata()
        tensor = torch.tensor(data, dtype=torch.float32)
        tensors.append(tensor)
    return torch.stack(tensors)

def load_adni_data(rank_worldsize:str,
                   adni_num:int,
                   data_dir:str,
                   img_dir:str,
                   csv_path:str,
                   csv_filename:str):
    """
    Load ADNI dataset and return tensors for train/test.
    """
    descriptor = MultiINPUTShardDescriptor(
        rank_worldsize=rank_worldsize,
        adni_num=adni_num,
        data_dir=data_dir,
        img_dir=img_dir,
        csv_path=csv_path,
        csv_filename=csv_filename,
    )

    train_meta = descriptor.data_by_type['train']
    test_meta = descriptor.data_by_type['val']

    base_dir = descriptor.data_dir
    duplicate_prefix = descriptor.data_dir.split('/')[-1]

    # Fix paths
    def fix_paths(meta):
        paths = []
        for path in meta['IMG_PATH_NORM_min-max'].tolist():
            if path.startswith(f'{duplicate_prefix}/'):
                path = os.path.join(base_dir, path.split(f'{duplicate_prefix}/', 1)[-1])
            else:
                path = os.path.join(base_dir, path)
            paths.append(path)
        return paths

    train_paths = fix_paths(train_meta)
    test_paths = fix_paths(test_meta)

    # Load images
    X_train = load_nifti_to_tensor(train_paths)
    y_train = torch.tensor(np.array(train_meta['labels'].tolist()), dtype=torch.long)

    X_test = load_nifti_to_tensor(test_paths)
    y_test = torch.tensor(np.array(test_meta['labels'].tolist()), dtype=torch.long)

    return (X_train, y_train), (X_test, y_test)

def get_dataset_info(dataset: str):
    if dataset == "mnist":
        input_size = (1, 32, 32)
        num_classes = 10
    elif dataset == "cifar10":
        input_size = (3, 32, 32)
        num_classes = 10
    elif dataset == "cifar100":
        input_size = (3, 32, 32)
        num_classes = 100
    elif dataset == "adni":
        input_size = (91, 109, 91)
        num_classes = 1 
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return input_size, num_classes

# ---------------------------------------------
# Updated load_data function
# ---------------------------------------------
def load_data(dataset: str, batch_size: int=128):
    num_workers = 2

    if dataset == "mnist":
        transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root='../data/', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='../data/', train=False, download=True, transform=transform)

    elif dataset == "cifar10":
        MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])
        train_set = datasets.CIFAR10('../data/cifar10/', train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR10('../data/cifar10/', train=False, transform=transform_test, download=True)

    elif dataset == "cifar100":
        MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)
        STD_CIFAR100 = (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100)])
        train_set = datasets.CIFAR100('../data/cifar100/', train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR100('../data/cifar100/', train=False, transform=transform_test, download=True)

    elif dataset == "adni":
        (X_train, y_train), (X_test, y_test) = load_adni_data(
            rank_worldsize='1,1',
            adni_num=1,
            data_dir='/mnt/shared_nfs/brunofolder/MERGE/WALTER/IMGS/a1',
            img_dir='ADNI1_ALL_T1',
            csv_path='/mnt/shared_nfs/brunofolder/MERGE/WALTER/IMGS/ADNI_csv',
            csv_filename='ADNI_ready.csv'
        )
        train_set = TensorDataset(X_train.unsqueeze(1), y_train)
        test_set = TensorDataset(X_test.unsqueeze(1), y_test)
        num_workers = 0  # large NIfTI files; safer with num_workers=0
        batch_size = 4

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Train/validation split
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader