#!/bin/bash

root='/data/shared/projects/PhectorDB/litpcba/TP53/pdb'
cdpkit='/data/shared/software/CDPKit-head-RH7/Bin'
python_scripts='/home/drose/git/PhectorDB/cdpkit_preprocessing/python_scripts/cdpl'

# Generate ligand-rezeptor pharmacophore
pdb_file='4ago'
ligand='P74'
python $python_scripts/pharm_gen_ia_ph4s.py -r $root/$pdb_file.pdb -l $root/$pdb_file.sdf -s $ligand -o $root/$pdb_file.pml