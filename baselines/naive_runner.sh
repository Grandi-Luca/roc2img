#!/bin/bash

datasets="mnist cifar10 cifar100"
models="minirocket multivariate_minirocket" #rocket multirocket multivariate_multirocket 
n_kernels_list="1000 2000 3000 4000 5000 6000" #5000 10000
channel_methods="row_wise column_wise " #zigzag spiral hilbert
seeds="2026 42 1234 0 9999"

alpha=0.1


#source ../.venv/bin/activate

for model in $models; do
    for n_kernels in $n_kernels_list; do
        for dataset in $datasets; do
            for channel_method in $channel_methods; do
                for seed in $seeds; do

                    echo "Running → model: $model | kernels: $n_kernels | dataset: $dataset | method: $channel_method | seed: $seed"
                    
                    python naive_r2i.py \
                        --model $model \
                        --n_kernels $n_kernels \
                        --dataset $dataset \
                        --channel_method $channel_method \
                        --alpha $alpha \
                        --seed $seed

                done
            done
        done
    done
done