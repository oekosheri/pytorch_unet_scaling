#!/usr/local_rwth/bin/zsh
#SBATCH --time=12:00:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tag_task
#SBATCH --account=*******


source ../environment/load_env_rocky.sh
source ../environment/horovod-env-rocky/bin/activate

${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c 'bash script.sh'

cp log.csv  ../logs/log_hvd_${SLURM_NTASKS}.csv
rm core.*


