#!/usr/local_rwth/bin/zsh

# clean env first
module purge

# Variant 1: GCC + Open MPI
# module load GCC/10.3.0 OpenMPI/4.1.1
# Variant 2: Intel + Intel MPI
module load intel-compilers/2021.2.0 impi/2021.6.0
module load GCCcore/.11.3.0

# load Python
module load Python/3.9.6
# load all CUDA related modules
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.12.12-CUDA-11.7.0
# CMake required for building Horovod
module load CMake/3.21.1

