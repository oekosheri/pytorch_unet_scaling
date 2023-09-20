#!/usr/local_rwth/bin/zsh
#SBATCH --time=20:30:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tag_task
#SBATCH --account=p0020572



source ../../environment/load_env_rocky.sh
source ../../environment/horovod-env-rocky/bin/activate

# module purge
# module load iimpi/2019b
# module list
# echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
# echo "R_WLM_ABAQUSHOSTLIST: ${R_WLM_ABAQUSHOSTLIST}"
echo "SLURMD_NODENAME: ${R_LOGINHOST}"
${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c 'source setup.sh  && bash script.sh ${R_LOGINHOST}'

# comm_1="${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c '\
# source setup.sh  && bash script.sh ${SLURMD_NODENAME}'"

# comm_2="source setup.sh && bash script.sh ${SLURMD_NODENAME}"


# command=tag_command


# if [ $command = 1 ]

# then

#     eval  "${comm_1}"

# else

#     eval  "${comm_2}"

# fi

# save the log file
cp log.csv  ../../logs/log_${SLURM_NTASKS}.csv
rm core.*

