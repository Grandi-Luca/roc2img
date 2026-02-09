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


class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden1=512, hidden2=256, num_classes=10):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        self.scheduler = None
    
    def forward(self, x):
        # Flatten dell'immagine: (batch, 3, 32, 32) -> (batch, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)
    

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion*planes)
      )
  
  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class BottleNeck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(BottleNeck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes , planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes :
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, num_blocks, in_channels, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512*block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion      
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

class ResNet18(ResNet):
    def __init__(self, in_channels, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[60,120],
            gamma=0.2
        )

    def forward(self, x):
        return super().forward(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.train()
    correct, total = 0, 0
    train_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criterion(outputs, targets)
        loss.backward()
        model.optimizer.step()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(loader.dataset)
    return 100. * correct / total, train_loss

def test_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    tot_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # calculate the batch loss
        tot_loss = model.criterion(outputs, targets)
        # update average loss 
        tot_loss += tot_loss.item()*inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # calculate average losses
    if loader.sampler is not None:
        tot_loss = tot_loss/len(loader.sampler)

    return 100. * correct / total, tot_loss

def train(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, device: torch.device, max_epochs=80, acc_target=0.0):

    train_time = 0.0
    valid_loss_min = np.inf

    for epoch in range(max_epochs):
        start_train = time.time()
        _, train_loss = train_epoch(model, train_loader, device)
        epoch_time = time.time() - start_train
        train_time += epoch_time
        valid_acc, valid_loss = test_epoch(model, valid_loader, device)
        
        print(f'Epoch {epoch+1}: Avg Train Loss: {train_loss:.4f} | Avg Valid Loss: {valid_loss:.4f} | Time: {epoch_time:.2f}s')
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pt')

        if acc_target > 0.0 and valid_acc >= acc_target:
            print(f'Target accuracy of {valid_acc}% reached. Stopping training.')
            break

        if model.scheduler is not None:
            model.scheduler.step()
    
    # Statistiche finali
    num_params = model.count_parameters()
    hours = int(train_time // 3600)
    minutes = int((train_time % 3600) // 60)
    seconds = int(train_time % 60)
    
    print(f'\n{"="*50}')
    print(f'STATISTICHE FINALI:')
    print(f'{"="*50}')
    print(f'Tempo di addestramento: {hours}h {minutes}m {seconds}s ({train_time:.2f} secondi totali)')
    print(f'Numero di parametri della rete: {num_params:,}')
    print(f'numero di epoche: {epoch+1}')

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.load_state_dict(torch.load(f'best_model_{model.__class__.__name__}.pt'))
    test_acc, _ = test_epoch(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'{"="*50}')
    print()


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

def get_model(model_name: str, input_size: tuple[int, ...], num_classes: int) -> nn.Module:
    if model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes, in_channels=input_size[0])
    elif model_name == 'mlp':
        input_size = input_size[0] * input_size[1] * input_size[2]
        model = MLP(input_size=input_size, hidden1=7000, hidden2=6000, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate models on CIFAR-10')
    parser.add_argument('--model', type=str, choices=['resnet18', 'mlp'], default='resnet18',
                        help='Model architecture to use (default: resnet18)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation (default: 128)')
    parser.add_argument('--max_epochs', type=int, default=160, help='Maximum number of training epochs (default: 160)')
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "cifar100"], default="cifar10",
                        help="Dataset to use (default: cifar10)")
    parser.add_argument('--acc_target', type=float, default=0.0, help='Target accuracy for early stopping (default: 0.0, no early stopping)')
    args = parser.parse_args()

    train_loader, valid_loader, test_loader = load_data(args.dataset, args.batch_size)

    num_classes = len(train_loader.dataset.classes)
    input_size = train_loader.dataset[0][0].shape

    model = get_model(args.model, input_size, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train(model, train_loader, valid_loader, device, max_epochs=args.max_epochs, acc_target=args.acc_target)
    evaluate(model, test_loader, device)