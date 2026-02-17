#!/bin/bash

datasets="mnist cifar10 cifar100 adni"
models="mlp resnet"
seeds="2026 42 1234 0 9999"

for dataset in $datasets; do
    for model in $models; do
        for seed in $seeds; do
            echo "Running with dataset: $dataset, model: $model, seed: $seed"
            python exp_run.py --dataset $dataset --model $model --seed $seed
        done
    done
done