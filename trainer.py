from pyexpat import features, model
from sched import scheduler
from unittest import loader

from zmq import device
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

from datasets import load_data
from models import get_model

class Trainer:

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
        self.model = self.model.to(self.device)
        self.optimizer, self.scheduler, self.num_epochs = self.__make_optimizer_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        

    def __make_optimizer_scheduler(self):
        
        if self.model.name.lower() == 'mlp':
            optimizer = optim.SGD(self.model.parameters(), lr=0.05)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[100,200],
                gamma=0.0
            )
            num_epochs = 300
        elif 'resnet' in self.model.name.lower():
            optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[60,120],
                gamma=0.2
            )
            num_epochs = 160
        elif 'vgg' in self.model.name.lower():
            optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[80,120],
                gamma=0.1
            )
            num_epochs = 160
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        return optimizer, scheduler, num_epochs
        

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        correct, total = 0, 0
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        return 100. * correct / total, train_loss


    def test_epoch(self, valid_loader: DataLoader):
        self.model.eval()
        correct, total = 0, 0
        tot_loss = 0.0
        
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            # calculate the batch loss
            tot_loss = self.criterion(outputs, targets)
            # update average loss 
            tot_loss += tot_loss.item()*inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # calculate average losses
        if valid_loader.sampler is not None:
            tot_loss = tot_loss/len(valid_loader.sampler)

        return 100. * correct / total, tot_loss


    def train(self, train_loader: DataLoader, valid_loader: DataLoader, acc_target=0.0):

        train_time = 0.0
        valid_loss_min = np.inf        

        for epoch in range(self.num_epochs):
            start_train = time.time()
            train_acc, train_loss = self.train_epoch(train_loader)
            epoch_time = time.time() - start_train
            train_time += epoch_time
            valid_acc, valid_loss = self.test_epoch(valid_loader)
            
            print(f'Epoch {epoch+1}: Avg Train Loss: {train_loss:.4f} | Avg Train Acc: {train_acc:.2f}% | Avg Valid Loss: {valid_loss:.4f} | Avg Valid Acc: {valid_acc:.2f}% | Time: {epoch_time:.2f}s')
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(
                valid_loss_min,
                valid_loss))
                valid_loss_min = valid_loss
                torch.save(self.model.state_dict(), f'best_model_{self.model.__class__.__name__}.pt')

            if acc_target > 0.0 and valid_acc >= acc_target:
                print(f'Target accuracy of {valid_acc}% reached. Stopping training.')
                break

            if self.scheduler is not None:
                self.scheduler.step()
        
        # Statistiche finali
        num_params = self.model.count_parameters()
        hours = int(train_time // 3600)
        minutes = int((train_time % 3600) // 60)
        seconds = int(train_time % 60)
        
        print(f'\n{"="*50}')
        print(f'STATISTICHE FINALI:')
        print(f'{"="*50}')
        print(f'Tempo di addestramento: {hours}h {minutes}m {seconds}s ({train_time:.2f} secondi totali)')
        print(f'Numero di parametri della rete: {num_params:,}')
        print(f'numero di epoche: {epoch+1}')

    def evaluate(self, test_loader: DataLoader):
        self.model.load_state_dict(torch.load(f'best_model_{self.model.__class__.__name__}.pt'))
        test_acc, _ = self.test_epoch(test_loader)
        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'{"="*50}')
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate models on CIFAR-10')
    parser.add_argument('--model', type=str, choices=['resnet18', 'mlp', 'vgg16'], default='resnet18',
                        help='Model architecture to use (default: resnet18)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation (default: 128)')
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "cifar100"], default="cifar10",
                        help="Dataset to use (default: cifar10)")
    parser.add_argument('--acc_target', type=float, default=0.0, help='Target accuracy for early stopping (default: 0.0, no early stopping)')
    args = parser.parse_args()

    train_loader, valid_loader, test_loader = load_data(args.dataset, args.batch_size)

    num_classes = len(train_loader.dataset.classes)
    input_size = train_loader.dataset[0][0].shape

    model = get_model(args.model, input_size, num_classes)
    
    trainer = Trainer(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    trainer.train(train_loader, valid_loader, acc_target=args.acc_target)
    trainer.evaluate(test_loader)