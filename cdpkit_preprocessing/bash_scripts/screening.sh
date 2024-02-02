#!/bin/bash

root='/data/shared/projects/PhectorDB/virtual_screening_ESR1_ant'
filename='inactives'
cdpkit='/data/shared/software/CDPKit-head-RH7/Bin'
python_scripts='/home/drose/git/PhectorDB/cdpkit_preprocessing/python_scripts'
files='/home/drose/git/PhectorDB/cdpkit_preprocessing/files'


# Alignment to reference pharmacophore
#python $python_scripts/pharm_align_mols.py -r $root/raw/query.pml -i $root/preprocessing/ligands.sdf -o $root/preprocessing/aligned.sdf -p

# Screening Database Generation
#$cdpkit/psdcreate -i $root/preprocessing/${filename}.sdf -o $root/raw/$filename.psd -d

# Show information of psd file
$cdpkit/psdinfo -i $root/raw/$filename.psd -C -P -F

# Screening Database Usage
$cdpkit/psdscreen -d $root/raw/$filename.psd -q $root/raw/queries.pml -o $root/vs/${filename}_aligned.sdf -S -I -C -D -m BEST-MATCH -x 0 -b -P -N

