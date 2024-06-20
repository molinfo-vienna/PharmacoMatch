#!/bin/bash

root='/data/shared/projects/PhectorDB'
dataset='training_data'
path="$root/$dataset"
filename="chembl_data"
cdpkit='/data/shared/software/CDPKit-head-RH7/Bin'
python_scripts='/home/drose/git/PhectorDB/data_processing/python_scripts'

mkdir $path/preprocessing
mkdir $path/processed
mkdir $path/raw

# Database clean-up, might need additional removal of duplicates after stripping
python $python_scripts/cdpl/clean_mol_db.py -i $path/input/$filename.smi -o $path/preprocessing/${filename}_clean.smi -s -c -d $path/preprocessing/${filename}_removed.smi -v 2

# Physiological pH
python $python_scripts/cdpl/prot_phys_cond.py -i $path/preprocessing/${filename}_clean.smi -o $path/preprocessing/${filename}_phys.smi 

# Duplicate removal
python $python_scripts/utils/remove_duplicates.py -i $path/preprocessing/${filename}_phys.smi -n 0

# Conformer Generation
$cdpkit/confgen -n 1 -t 100 -T 0 -i $path/preprocessing/${filename}_phys.smi  -o $path/preprocessing/$filename.sdf

# Pharmacophore Generation
python $python_scripts/cdpl/pharm_gen_mol_ph4s.py -i $path/preprocessing/$filename.sdf -o $path/raw/$filename.cdf

