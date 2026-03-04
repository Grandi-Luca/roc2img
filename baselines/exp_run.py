import torch
import torch.nn as nn
import numpy as np
from torchvision import models

import random
import os

import argparse
import time

from datasets import load_data, get_dataset_info
from trainer import Trainer
from logger import Logger, WandbLogger


from models.mlp import MLP, MLPAdni
from models.resnet import ResNet, ResNet3D
from models.vgg import VGG, VGG3D
from models.mobilenetv4 import mobilenetv4_conv_small
from models.pcanet import build_pcanet_model


def get_model(model_name: str, dataset_name: str, input_size: tuple[int, ...], num_classes: int) -> nn.Module:
    
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    if 'resnet' in model_name:
        if dataset_name == 'adni':
            model = ResNet3D(
                name=model_name,
                n_input_channels=1,
                widen_factor=1.0,
                n_classes=num_classes
            )
        else:
            model = ResNet(
                name=model_name,
                num_classes=num_classes,
                in_channels=input_size[0]
            )

    elif model_name == 'mlp':
        flatten_size = 1
        for dim in input_size:
            flatten_size *= dim
        
        if dataset_name == 'adni':
            model = MLPAdni(
                input_shape=input_size,
                num_classes=num_classes
            )
        else:
            model = MLP(
                input_size=flatten_size,
                num_classes=num_classes
            )

    elif 'vgg' in model_name:
        if dataset_name == 'adni':
            model = VGG3D(
                name=model_name,
                in_channels=1,
                num_classes=num_classes,
                batch_norm=True
            )
        else:
            model = VGG(
                name=model_name,
                num_classes=num_classes,
                in_channels=input_size[0],
                batch_norm=True
            )

    elif 'mobilenet' in model_name:
        model = mobilenetv4_conv_small(
            dataset_name=dataset_name,
            name=model_name,
            num_classes=num_classes,
            in_channels=1 if dataset_name == 'adni' else input_size[0]
        )
    
    elif 'pcanet' in model_name:
        model = build_pcanet_model(dataset_name=dataset_name)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate models on CIFAR-10')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture to use (default: resnet18)')
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset to use (default: cifar10)")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    set_random_seed(args.seed)
    
    train_loader, valid_loader, test_loader = load_data(args.dataset)
    
    input_size, num_classes = get_dataset_info(args.dataset)
    model = get_model(args.model, args.dataset, input_size, num_classes)
    
    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
        
    logger = WandbLogger(name=run_name, 
                         entity="mlgroup",
                         project="roc2img",
                         logs_directory="./logs/", 
                         results_directory="./checkpoints/", 
                         log_metrics_directory="./log_metrics/",
                         tags=[ args.model,
                                args.dataset,
                                f"seed{args.seed}",
                                "baseline"]
                         )
    
    trainer = Trainer(model, 
                      dataset_name=args.dataset,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                      logger=logger)
    
    config = {
            'model': trainer.model.__class__.__name__,
            'dataset': args.dataset,
            'num_classes': num_classes,
            'optimizer': trainer.optimizer.__class__.__name__,
            'scheduler': trainer.scheduler.__class__.__name__ if trainer.scheduler is not None else None,
            'num_epochs': trainer.num_epochs
        }
        
    logger.log(config, header="Configuration")
    
    trainer.start_train(train_loader, valid_loader, test_loader)
    
    logger.finish()