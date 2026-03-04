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

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import numpy as np
from logger import WandbLogger
import time

class Trainer:

    def __init__(self, model: nn.Module, dataset_name: str, device: torch.device, logger: WandbLogger = None):
        self.model = model
        self.logger = logger
        self.device = device
        self.dataset_name = dataset_name
        
        self.optimizer, self.scheduler, self.num_epochs , self.criterion = None, None, None, None
        
        

    def __make_optimizer(self):
                
        if 'mlp' in self.model.name.lower():
            if self.dataset_name.lower() == 'adni':
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=1e-3,
                    weight_decay=1e-3
                )
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=100
                )
                
                num_epochs = 100
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=0.05)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[100,200],
                    gamma=0.1
                )
                num_epochs = 300
            
        elif 'resnet' in self.model.name.lower():
            
            if self.dataset_name.lower() == 'adni':
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=120
                )
                num_epochs = 120
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[60,120],
                    gamma=0.2
                )
                num_epochs = 160
        elif 'vgg' in self.model.name.lower():
            
            if self.dataset_name.lower() == 'adni':
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=120
                )
                num_epochs = 120
            elif self.dataset_name.lower() == 'mnist':
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=0.05,
                    momentum=0.9,
                    weight_decay=5e-4
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[30, 45],
                    gamma=0.1
                )
                num_epochs = 60
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[80,120],
                    gamma=0.1
                )
                num_epochs = 160
        elif 'mobilenet' in self.model.name.lower():
            if self.dataset_name.lower() == 'adni':
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=2e-4,
                    weight_decay=1e-4
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=120
                )

                num_epochs = 120

            elif self.dataset_name.lower() == 'cifar100':
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=5e-4
                )

                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[150, 225],
                    gamma=0.1
                )

                num_epochs = 300

            elif self.dataset_name.lower() == 'cifar10':
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=5e-4
                )

                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[120, 180],
                    gamma=0.1
                )

                num_epochs = 240

            elif self.dataset_name.lower() == 'mnist':
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=1e-3,
                    weight_decay=1e-4
                )

                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=20,
                    gamma=0.5
                )

                num_epochs = 60

            else:
                raise ValueError(f"Unsupported dataset for MobileNet: {self.dataset_name}")
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        
        if self.dataset_name.lower() == 'adni':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        correct, total = 0, 0
        train_loss = 0.0
        
        scaler = torch.cuda.amp.GradScaler()  # mixed precision scaler
        
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                
                if self.dataset_name.lower() == 'adni':
                    targets = targets.float().unsqueeze(1)  # BCE expects shape [B,1]
                
                loss = self.criterion(outputs, targets)
            
            # backward and optimizer step with mixed precision
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # accumulate statistics
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            
            if self.dataset_name.lower() == 'adni':
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += batch_size

        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        return train_acc, train_loss


    def test_epoch(self, valid_loader: DataLoader):
        self.model.eval()
        correct, total = 0, 0
        tot_loss = 0.0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                
                if self.dataset_name.lower() == 'adni':
                    targets = targets.float().unsqueeze(1)
                
                batch_loss = self.criterion(outputs, targets)
                tot_loss += batch_loss.item() * inputs.size(0)
                
                if self.dataset_name.lower() == 'adni':
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, predicted = outputs.max(1)
                correct += (predicted == targets).sum().item()
                total += inputs.size(0)
        
        tot_loss /= len(valid_loader.dataset)
        valid_acc = 100. * correct / total

        return valid_acc, tot_loss


    def train(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader):

        train_time = 0.0
        valid_loss_min = np.inf
        
        self.model = self.model.to(self.device)
        self.__make_optimizer()        

        for epoch in range(self.num_epochs):
            start_train = time.time()
            train_acc, train_loss = self.train_epoch(train_loader)
            epoch_time = time.time() - start_train
            train_time += epoch_time
            valid_acc, valid_loss = self.test_epoch(valid_loader)
            test_acc, test_loss = self.test_epoch(test_loader)
                        
            self.logger({
                'epoch': epoch+1,
                'train loss': train_loss,
                'train acc': train_acc,
                'valid acc': valid_acc,
                'valid loss': valid_loss,
                'test acc': test_acc,
                'test loss': test_loss,
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
        self.logger({
                'num params': num_params
            })
        hours = int(train_time // 3600)
        minutes = int((train_time % 3600) // 60)
        seconds = int(train_time % 60)
        
        self.logger.log(f'Training completed in {hours}h {minutes}m {seconds}s with {num_params:,} parameters over {epoch+1} epochs.')
    
    def get_x_y_from_loader(self, loader: DataLoader):
        X, y = [], []
        for inputs, targets in loader:
            X.append(inputs.permute(0, 2, 3, 1).cpu().numpy())
            y.append(targets.cpu().numpy())
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        return X, y
    
    def train_pca_net(self, train_loader, valid_loader, test_loader):
        
        with torch.no_grad():
            X_train, y_train = self.get_x_y_from_loader(train_loader)
            X_test, y_test = self.get_x_y_from_loader(test_loader)

            # ------------------
            # Train PCANet
            # ------------------
            start_time = time.time()
            self.logger.log("Fiiting PCANet...")
            self.model.fit(X_train)
            train_time = time.time() - start_time

            # ------------------
            # Transform train
            # ------------------
            start_time = time.time()
            self.logger.log("Transforming train set with PCANet...")
            X_train_transf = self.model.transform(X_train)
            transform_time = time.time() - start_time

            # ------------------
            # Train Linear SVM (paper-consistent)
            # ------------------
            start_time = time.time()
            self.logger.log("Training Linear SVM on PCANet features...")
            classifier = LinearSVC(C=10)
            classifier.fit(X_train_transf, y_train)
            classifier_time = time.time() - start_time

            # ------------------
            # Train Accuracy
            # ------------------
            self.logger.log("Evaluating train set predictions...")
            train_pred = classifier.predict(X_train_transf)
            train_acc = accuracy_score(y_train, train_pred)

            # ------------------
            # Test
            # ------------------
            start_time = time.time()
            self.logger.log("Transforming test set with PCANet...")
            X_test_transf = self.model.transform(X_test)
            test_transform_time = time.time() - start_time

            self.logger.log("Evaluating test set predictions...")
            y_pred = classifier.predict(X_test_transf)
            test_acc = accuracy_score(y_test, y_pred)

            total_time = train_time + transform_time + classifier_time + test_transform_time

            self.logger({
                'train time': train_time,
                'transform time': transform_time,
                'classifier time': classifier_time,
                'test transform time': test_transform_time,
                'total time': total_time,
                'train acc': train_acc,
                'test acc': test_acc
            })
    
    
    def start_train(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader):
        if self.model.name.lower() != 'pcanet':
            self.train(train_loader, valid_loader, test_loader)
        else:
            self.train_pca_net(train_loader, valid_loader, test_loader)
            

    def evaluate(self, test_loader: DataLoader):
        self.model.load_state_dict(torch.load(f'checkpoints/best_model_{self.model.__class__.__name__}.pt'))
        self.model = self.model.to(self.device)
        test_acc, _ = self.test_epoch(test_loader)
        self.logger({'test acc': test_acc})