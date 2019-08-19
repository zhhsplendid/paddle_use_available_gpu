#!/bin/sh

export ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"


set -ex
echo "Root work path  $ROOT_PATH"
echo "Use cuda device $CUDA_VISIBLE_DEVICES"

nvcc $ROOT_PATH/take_cuda_mem.cu -o $ROOT_PATH/take_cuda_mem.o -std=c++11 -lpthread

# Start a program taking 4 GB GPU
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"} nohup $ROOT_PATH/take_cuda_mem.o 1>/dev/null 2>/dev/null &

# PaddlePaddle takes about 10 GB GPU
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"} python create_paddle_data.py

# Kill the 4GB program after PaddlePaddle success
$ROOT_PATH/kill.sh
