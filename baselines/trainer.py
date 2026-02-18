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
from logger import WandbLogger
import time

class Trainer:

    def __init__(self, model: nn.Module, device: torch.device, logger: WandbLogger = None):
        self.model = model
        self.logger = logger
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
        train_acc = 100. * correct / total
        
        return train_acc, train_loss


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


    def train(self, train_loader: DataLoader, valid_loader: DataLoader):

        train_time = 0.0
        valid_loss_min = np.inf        

        for epoch in range(self.num_epochs):
            start_train = time.time()
            train_acc, train_loss = self.train_epoch(train_loader)
            epoch_time = time.time() - start_train
            train_time += epoch_time
            valid_acc, valid_loss = self.test_epoch(valid_loader)
                        
            self.logger({
                'epoch': epoch+1,
                'train loss': train_loss,
                'train acc': train_acc,
                'valid acc': valid_acc,
                'train time': epoch_time
            })

            if valid_loss <= valid_loss_min:
                
                self.logger.log(f"Saving model with new min validation loss: {valid_loss:.6f} at epoch: {epoch+1}")
                valid_loss_min = valid_loss
                torch.save(self.model.state_dict(), f'checkpoints/best_model_{self.model.__class__.__name__}.pt')

            if self.scheduler is not None:
                self.scheduler.step()
        
        # Statistiche finali
        num_params = self.model.count_parameters()
        hours = int(train_time // 3600)
        minutes = int((train_time % 3600) // 60)
        seconds = int(train_time % 60)
        
        self.logger.log(f'Training completed in {hours}h {minutes}m {seconds}s with {num_params:,} parameters over {epoch+1} epochs.')
        

    def evaluate(self, test_loader: DataLoader):
        self.model.load_state_dict(torch.load(f'checkpoints/best_model_{self.model.__class__.__name__}.pt'))
        test_acc, _ = self.test_epoch(test_loader)
        self.logger({'test acc': test_acc})