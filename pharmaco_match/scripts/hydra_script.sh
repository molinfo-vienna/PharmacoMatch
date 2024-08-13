#$ -S /bin/bash                                 # use the bash 
#$ -M daniel.rose@univie.ac.at                  # send an email after the job is finished
#$ -m e                                         # send an e-mail if an error occurs. 
#$ -j y                                         # not sure ?
#$ -p -700                                      # set the priority
#$ -pe smp 1                                    # use a single CPU on a shared memory instance
#$ -o /data/cluster/logs_DR                     # append output/error to a log file and write the log file 
#$ -l h=node02                                  # submit to node02
#$ -l gpu=1                                     # 1 GPUs are required
#$ -q gpu.q                                     # submit to the GPU queue

conda activate ph4
python3 /home/drose/git/PhectorDB/pharmaco_match/scripts/training.py 2