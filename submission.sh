localDir=`pwd`
run_file=$localDir/run_file.sh
submit_file=$localDir/submit_file.sh
program=$localDir/training.py
setup=$localDir/setup_dist_env.sh


epochs=150
bs=16
name="Indents_"


# rm -r $localDir/Logs
# mkdir $localDir/Logs

for gpu in   1 2 4 6 8 10 12 14
do

    for cpu in 12
    do

        for augment in 0
        do


            mkdir -p $localDir/$name$gpu$augment$cpu
            cd $localDir/$name$gpu$augment$cpu


            if [ $gpu = 1 ]
            then
                batch=$bs
                tasks=1
                command=2
                node=1
            else
                batch=$((${gpu}*bs))
                tasks=2
                command=1
                node=$((${gpu}/2))

            fi



            # adapting run file
            sed -e "s|tag_program|${program}|g" ${run_file}  |\
            sed -e "s/\<tag_epoch\>/${epochs}/g"| \
            sed -e "s/\<tag_batch\>/${batch}/g"| \
            sed -e "s/\<tag_aug\>/${augment}/g" > script.sh

            # adapting submit file
            sed -e "s/\<tag_task\>/${tasks}/g" ${submit_file}|\
            sed -e "s/\<tag_node\>/${node}/g" | \
            sed -e "s/\<tag_cpu\>/${cpu}/g" | \
            sed -e "s|setup.sh|${setup}|g" | \
            sed -e "s/\<tag_command\>/${command}/g" > sub_${node}.sh
            sbatch sub_${node}.sh

        done
    done
done



