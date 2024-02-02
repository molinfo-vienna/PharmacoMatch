#!/bin/bash

root='/data/shared/projects/PhectorDB/virtual_screening_ESR1_ant'
filename='inactives'
cdpkit='/data/shared/software/CDPKit-head-RH7/Bin'
python_scripts='/home/drose/git/PhectorDB/cdpkit_preprocessing/python_scripts'
files='/home/drose/git/PhectorDB/cdpkit_preprocessing/files'

# Generate ligand-rezeptor pharmacophore
pdb_file='1xp1'
ligand='AIH'
python $python_scripts/pharm_gen_ia_ph4s.py -r $files/$pdb_file.pdb -l $files/$pdb_file.sdf -s $ligand -o $files/$pdb_file.pml