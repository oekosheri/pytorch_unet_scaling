#!/usr/local_rwth/bin/zsh

# for regular environment
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
