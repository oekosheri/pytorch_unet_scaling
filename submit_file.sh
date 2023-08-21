#!/usr/local_rwth/bin/zsh
#SBATCH --time=12:00:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=tag_cpu
#SBATCH --gres=gpu:tag_task
#SBATCH --account=p0020572




source ../environment/load_env_rocky.sh
source ../environment/horovod-env-rocky/bin/activate


comm_1="${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c 'bash script.sh'"

comm_2="bash script.sh"


command=tag_command


if [ $command = 1 ]

then

    eval  "${comm_1}"

else

    eval  "${comm_2}"

fi

# save the log file

cp log.csv  ../Logs/log_hvd_${SLURM_NTASKS}.csv



