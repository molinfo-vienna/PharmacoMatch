#!/bin/bash

cdpkit='/data/shared/software/CDPKit-head-RH9/Bin'
project='/data/sharedXL/projects/PharmacoMatch'
path="$project/Enamine_HLL"
filename="enamine_hll"
python_scripts="/home/drose/git/PharmacoMatch/data_processing/python_scripts"

mkdir $path/preprocessing
mkdir $path/processed
mkdir $path/raw

# Database clean-up, might need additional removal of duplicates after stripping
#python $python_scripts/cdpl/clean_mol_db.py -i $path/input/$filename.sdf -o $path/preprocessing/${filename}_clean.smi -s -c -d $path/preprocessing/${filename}_removed.smi -v 2

# Physiological pH
#python $python_scripts/cdpl/prot_phys_cond.py -i $path/preprocessing/${filename}_clean.smi -o $path/preprocessing/${filename}_phys.smi 

# Removal of duplicates and compounds that might occur in the test benchmark dataset
python $python_scripts/utils/remove_duplicates.py -i $path/preprocessing/${filename}_phys.smi 

# Conformer Generation
$cdpkit/confgen -n 25 -t 100 -T 0 -i $path/preprocessing/${filename}_phys.smi  -o $path/preprocessing/${filename}_filtered.sdf

# Pharmacophore Generation
$cdpkit/psdcreate -i $path/preprocessing/${filename}_filtered.sdf -o $path/raw/$filename.psd -d


