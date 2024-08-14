#!/bin/bash

cdpkit='<your_path>/CDPKit/Bin'
project='<your_path>/PharmacoMatch'
path="$project/data/training_data"
filename="chembl_data"
python_scripts="$project/data_processing/python_scripts"

mkdir $path/preprocessing
mkdir $path/processed
mkdir $path/raw

# Database clean-up, might need additional removal of duplicates after stripping
python $python_scripts/cdpl/clean_mol_db.py -i $path/input/$filename.smi -o $path/preprocessing/${filename}_clean.smi -s -c -d $path/preprocessing/${filename}_removed.smi -v 2

# Physiological pH
python $python_scripts/cdpl/prot_phys_cond.py -i $path/preprocessing/${filename}_clean.smi -o $path/preprocessing/${filename}_phys.smi 

# Removal of duplicates and compounds that might occur in the test benchmark dataset
python $python_scripts/utils/remove_test_data_from_training_data.py -i $path/preprocessing/${filename}_phys.smi -o $path/preprocessing/${filename}_filtered.smi -t $benchmark_root

# Conformer Generation
$cdpkit/confgen -n 1 -t 100 -T 0 -i $path/preprocessing/${filename}_filtered.smi  -o $path/preprocessing/${filename}_filtered.sdf

# Pharmacophore Generation
python $python_scripts/cdpl/pharm_gen_mol_ph4s.py -i $path/preprocessing/${filename}_filtered.sdf -o $path/raw/${filename}_filtered.cdf

