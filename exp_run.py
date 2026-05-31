import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from rocketnet import RocketNet, RocketLayer
from utils import AAP, GeneralizedMeanPooling, AGeM

STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)

def load_cifar10_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    # Preprocess the CIFAR-10 train dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
            transforms.Lambda(lambda x: x.unsqueeze(1)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root='./data/cifar10/', train=True, download=True, transform=transform)

    test_set = torchvision.datasets.CIFAR10(
        root='./data/cifar10/', train=False, download=True, transform=transform)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
        num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        num_workers=2)

    return train_loader, test_loader

def train_ridge_classfier(alpha, model, train_loader: DataLoader, test_loader: DataLoader) -> float:
    X_train, y_train, time_to_trasform_train = transform_inputs(train_loader, model)
    X_test, y_test, time_to_trasform_test = transform_inputs(test_loader, model)

    clf = RidgeClassifier(alpha).fit(X_train, y_train)

    test_score = accuracy_score(y_test, clf.predict(X_test))
    train_score = accuracy_score(y_train, clf.predict(X_train))
    print(f'Train score: {train_score}\t Test_score: {test_score}')

    del X_train, y_train, X_test, y_test
    return round(time_to_trasform_train, 3), round(time_to_trasform_test, 3), train_score, test_score


def transform_inputs(loader: DataLoader, model: RocketNet) -> tuple[np.ndarray, np.ndarray, float]:
    X = []
    y = []

    start = time.time()
    for inputs, target in tqdm(loader, desc="Transforming inputs"):
        X.append(model(inputs.to(model.device)).cpu())
        y.append(target)
    transform_time = time.time() - start

    return torch.cat(X, dim=0).numpy(), torch.cat(y, dim=0).numpy(), transform_time


if __name__ == "__main__":

    train_loader, test_loader = load_cifar10_data(128)

    input_shape = train_loader.dataset[0][0].shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_dim = min(input_shape[-2:])
    for i in range(3):
        current_block_layers = [
            (
                RocketLayer,
                input_shape[0] if j==0 else 1024,
                1024,
                3,
                1,
                True,
                input_shape[-2:] if j==0 else (min_dim//(2*(j+1)), min_dim//(2*(j+1))),
                [AAP(min_dim//(2*(j+1)) if j<i else 2)],
                device
            )
            for j in range(i+1)
        ]
        current_block_layers.append(nn.Flatten())
        model = RocketNet(device, 0, current_block_layers)

        print(f'Model {i+1}')
        print(model)
        train_ridge_classfier(0.1, model, train_loader, test_loader)
        print()
