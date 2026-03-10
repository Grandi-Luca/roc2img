#!/bin/bash

datasets="adni" #mnist cifar10 cifar100 
n_kernels_list="1000" #1000 2000 3000 4000 5000 6000
channel_methods="row_wise" #column_wise zigzag spiral hilbert
channel_handlings="separate"
scalers="true"
seeds="2026 42 1234 0 9999" #2026 42 1234 0 9999

source ../.venv/bin/activate

for n_kernels in $n_kernels_list; do
    for dataset in $datasets; do
        for channel_method in $channel_methods; do
            for channel_handling in $channel_handlings; do
                for scaler in $scalers; do
                    for seed in $seeds; do

                        echo "Running → model: ROCKET | kernels: $n_kernels | dataset: $dataset | method: $channel_method | handling: $channel_handling | seed: $seed"
                        
                        python3 naive_r2i.py \
                            --n_kernels $n_kernels \
                            --dataset $dataset \
                            --channel_method $channel_method \
                            --channel_handling $channel_handling \
                            --seed $seed \
                            --scaler $scaler

                    done
                done
            done
        done
    done
done

