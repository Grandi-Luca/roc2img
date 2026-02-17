from pyexpat import features
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


def get_model(model_name: str, input_size: tuple[int, ...], num_classes: int) -> nn.Module:
    if model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes, in_channels=input_size[0])
    elif model_name == 'mlp':
        input_size = input_size[0] * input_size[1] * input_size[2]
        model = MLP(input_size=input_size,num_classes=num_classes)
    elif 'vgg' in model_name.lower():
        model = VGG(num_classes=num_classes, in_channels=input_size[0], vgg_type=model_name.lower(), batch_norm=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_dim=1024, num_classes=10, num_hidden_layers=4):
        super(MLP, self).__init__()
        self.name = 'mlp'
        self.first_layer = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, hidden_dim),
                    nn.ReLU(),
        )
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_hidden_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, num_classes, bias=False)
        self.network = nn.Sequential(
            self.first_layer,
            *self.hidden_layers,
            self.output_layer
        )
    
    def forward(self, x):
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
        self.name = 'resnet18'
        
    def forward(self, x):
        return super().forward(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes=10, vgg_type='vgg16', batch_norm=False):
        super(VGG, self).__init__()
        self.cfgs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        if vgg_type not in self.cfgs:
            raise ValueError(f"Unsupported VGG type: {vgg_type}")

        self.name = vgg_type
        features = nn.Sequential(*self.__make_vgg_layers(batch_norm=batch_norm, in_channels=in_channels))
        classifier = nn.Sequential(*[
            nn.Sequential(*[nn.Flatten(), nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout()]),
            nn.Sequential(*[nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout()]),
            nn.Linear(4096, num_classes)
        ])
        
        self.network = nn.Sequential(*[features, classifier])
        
    
    def __make_vgg_layers(self, batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
        layers = []
        for ind,v in enumerate(self.cfgs[self.name]):
            if v == "M":
                continue
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    if self.cfgs[self.name][ind + 1] == "M":
                        layers += [nn.Sequential(*[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])]
                    else:
                        layers += [nn.Sequential(*[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])]
                else:
                    if self.cfgs[self.name][ind + 1] == "M":
                        layers += [nn.Sequential(*[conv2d, nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])]
                    else:
                        layers += [nn.Sequential(*[conv2d, nn.ReLU(inplace=True)])]
                in_channels = v
        return layers

    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)