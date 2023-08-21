#!/usr/local_rwth/bin/zsh

# for regular environment
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
# export MASTER_ADDR=${SLURMD_NODENAME}
# export MASTER_PORT=29500
# export PYTORCH_INIT_FILE="pytorch_init_${SLURM_NNODES}nodes_${SLURM_NTASKS}tasks_${SLURM_NTASKS_PER_NODE}tpn.pi"

