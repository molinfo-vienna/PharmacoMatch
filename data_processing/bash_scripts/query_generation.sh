#!/bin/bash
dataset='DUDE-Z'
target='UROK'
pdb_file='1sqt'
ligand='UI3'
root="/data/shared/projects/PhectorDB/$dataset/$target/pdb"
cdpkit='/data/shared/software/CDPKit-head-RH7/Bin'
python_scripts='/home/drose/git/PhectorDB/data_processing/python_scripts/cdpl'

# Generate ligand-rezeptor pharmacophore
python $python_scripts/pharm_gen_ia_ph4s.py -r $root/$pdb_file.pdb -l $root/$pdb_file.sdf -s $ligand -o $root/$pdb_file.pml