import torch
import torch.nn as nn
import torch.nn.functional as F


######################################
# VGG 2D
######################################

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

######################################################
# VGG 3D
#####################################################

class VGG3D(nn.Module):
    def __init__(self,
                 name,
                 in_channels,
                 num_classes=400,
                 batch_norm=False):

        super(VGG3D, self).__init__()

        self.cfgs = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                      512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                      512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        if name not in self.cfgs:
            raise ValueError(f"Unsupported VGG type: {name}")

        self.name = name

        self.features = nn.Sequential(
            *self._make_vgg3d_layers(batch_norm=batch_norm,
                                     in_channels=in_channels)
        )

        # Adaptive pooling makes it input-size agnostic
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def _make_vgg3d_layers(self,
                           batch_norm: bool = False,
                           in_channels: int = 3):

        layers = []

        for v in self.cfgs[self.name]:
            if v == 'M':
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            else:
                conv3d = nn.Conv3d(in_channels,
                                   v,
                                   kernel_size=3,
                                   padding=1)

                if batch_norm:
                    layers.extend([
                        conv3d,
                        nn.BatchNorm3d(v),
                        nn.ReLU(inplace=True)
                    ])
                else:
                    layers.extend([
                        conv3d,
                        nn.ReLU(inplace=True)
                    ])

                in_channels = v

        return layers

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)