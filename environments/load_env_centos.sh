#!/usr/local_rwth/bin/zsh

module purge
module load DEVELOP     \
    gcc/9               \
    cuda/11.4           \
    cudnn/8.3.2         \
    nccl/2.10.3         \
    cmake/3.21.1        \
    intelmpi/2018       \
    python/3.9.6

