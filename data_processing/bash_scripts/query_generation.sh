#!/bin/bash

root='/data/sharedXL/projects/PharmacoMatch/DUD-E'
ligand_suffix='sdf'

for target in "$root"/*; do
    if [ -d "$target" ]; then
        dir_path="$target/pdb"
        pdb_file=$(find "$dir_path" -maxdepth 1 -type f -name "*.pdb")
        ligand_file=$(find "$dir_path" -maxdepth 1 -type f -name "*.$ligand_suffix")
        mkdir $target/raw

        if [[ -n "$ligand_file" ]]; then
            lettercode=$(basename "$ligand_file" .$ligand_suffix)
        else
            echo "No .$ligand_suffix file found in the directory."
            exit 1
        fi
        output_path="$target/raw/query_full.pml"
        python3 /home/drose/git/PharmacoMatch/data_processing/python_scripts/cdpl/pharm_gen_ia_ph4s.py -r $pdb_file -l $ligand_file -o $output_path -s ${lettercode}
        echo ${lettercode}
    fi
done


