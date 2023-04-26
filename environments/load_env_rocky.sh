#!/usr/local_rwth/bin/zsh

# clean env first
module purge

# Variant 1: GCC + Open MPI
# module load GCC/10.3.0 OpenMPI/4.1.1
# Variant 2: Intel + Intel MPI
module load intel-compilers/2021.2.0 impi/2021.6.0

# load Python
module load Python/3.9.6
# load all CUDA related modules
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load NCCL/2.10.3-CUDA-11.3.1
# CMake required for building Horovod
module load CMake/3.21.1