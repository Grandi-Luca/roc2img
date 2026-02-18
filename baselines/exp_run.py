import torch
import numpy as np
from torchvision import models

import random
import os

import argparse
import time

from datasets import load_data
from models import get_model
from trainer import Trainer
from logger import Logger, WandbLogger


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

    num_classes = len(train_loader.dataset.classes)
    input_size = train_loader.dataset[0][0].shape

    model = get_model(args.model, input_size, num_classes)
    
    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
        
    logger = WandbLogger(name=run_name, 
                         entity="mlgroup",
                         project="roc2img",
                         logs_directory="./logs/", 
                         results_directory="./checkpoints/", 
                         log_metrics_directory="./log_metrics/")
    
    trainer = Trainer(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), logger=logger)
    
    config = {
            'model': trainer.model.__class__.__name__,
            'dataset': args.dataset,
            'num_classes': num_classes,
            'optimizer': trainer.optimizer.__class__.__name__,
            'scheduler': trainer.scheduler.__class__.__name__ if trainer.scheduler is not None else None,
            'num_epochs': trainer.num_epochs
        }
        
    logger.log(config, header="Configuration")
    
    trainer.train(train_loader, valid_loader)
    trainer.evaluate(test_loader)
    
    logger.finish()