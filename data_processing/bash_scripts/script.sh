#!/bin/bash

vars=('active_T' 'inactive_T' 'active_V' 'inactive_V')
datasets=('BACE')
#'TP53' 'ESR1_ant' 
for dataset in ${datasets[@]};
do
    root='/home/drose/LIT-PCBA/'$dataset
    target_dir=$root/'processed'
    mkdir $target_dir
    mkdir $target_dir/sdf
    mkdir $target_dir/cdf
    mkdir $target_dir/csv
    for var in ${vars[@]}; 
    do
        /data/shared/software/CDPKit-head-RH7/Bin/confgen -W 1 -t 256 -i $root/$var.smi -o $target_dir/sdf/$var.sdf
        python pharm_gen_mol_ph4s.py -i $target_dir/sdf/$var.sdf -o $target_dir/cdf/$var.cdf
        python pharm2graph2.py $target_dir/cdf/$var.cdf
        mv $target_dir/cdf/$var.csv $target_dir/csv/$var.csv
    done
done