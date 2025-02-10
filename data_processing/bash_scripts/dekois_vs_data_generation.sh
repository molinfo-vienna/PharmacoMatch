#!/bin/bash

# This script implements a pipeline to transform SMILES data into the ScreeningDatabase format
#root='/data/sharedXL/projects/PharmacoMatch/DUD-E'
#python_scripts='/home/drose/git/PharmacoMatch/data_processing/python_scripts'
#cdpkit='/data/shared/software/CDPKit-head-RH9/Bin'

root='<your_path>/PharmacoMatch/data/DUD-E'
python_scripts='<your_path>/PharmacoMatch/data_processing/python_scripts'
cdpkit='<your_path>/CDPKit-head-RH9/Bin'

filenames=('actives' 'inactives')

for target in "$root"/*; do
    if [ -d "$target" ]; then
        echo "Processing $target"
        path="$target"
        mkdir $path/preprocessing
        mkdir $path/processed
        mkdir $path/raw

        for filename in ${filenames[@]};
        do
            # Database clean-up, might need additional removal of duplicates after stripping
            python $python_scripts/cdpl/clean_mol_db_modified.py -i $path/input/$filename.smi -o $path/preprocessing/${filename}_clean.smi -s -c -d $path/preprocessing/${filename}_removed.smi -v 2

            # Physiological pH
            python $python_scripts/cdpl/prot_phys_cond_modified.py -i $path/preprocessing/${filename}_clean.smi -o $path/preprocessing/${filename}_phys.smi 

            # Duplicate removal
            python $python_scripts/utils/remove_duplicates.py -i $path/preprocessing/${filename}_phys.smi 

            # Conformer Generation
            $cdpkit/confgen -n 25 -t 100 -T 0 -i $path/preprocessing/${filename}_phys.smi  -o $path/preprocessing/$filename.sdf

            # Pharmacophore Generation
            $cdpkit/psdcreate -i $path/preprocessing/${filename}.sdf -o $path/raw/$filename.psd -d
        done
    fi
done