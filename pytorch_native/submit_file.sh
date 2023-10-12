#!/usr/local_rwth/bin/zsh
#SBATCH --time=20:30:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tag_task
#SBATCH --account=*******



source ../../environment/load_env_rocky.sh
source ../../environment/horovod-env-rocky/bin/activate


echo "SLURMD_NODENAME: ${R_LOGINHOST}"
${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c 'source setup.sh  && bash script.sh ${R_LOGINHOST}'


# save the log file
cp log.csv  ../../logs/log_${SLURM_NTASKS}.csv
rm core.*

