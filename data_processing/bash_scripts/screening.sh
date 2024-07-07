#!/bin/bash

root='/data/shared/projects/PhectorDB/DUD-E/UROK'
filename='inactives'
cdpkit='/data/shared/software/CDPKit-head-RH7/Bin'


# Alignment to reference pharmacophore
#python $python_scripts/pharm_align_mols.py -r $root/raw/query.pml -i $root/preprocessing/ligands.sdf -o $root/preprocessing/aligned.sdf -p

# Screening Database Generation
#$cdpkit/psdcreate -i $root/preprocessing/${filename}.sdf -o $root/raw/$filename.psd -d

# Show information of psd file
$cdpkit/psdinfo -i $root/raw/$filename.psd -C -P -F

# Screening Database Usage
start=$SECONDS
$cdpkit/psdscreen -d $root/raw/$filename.psd -q $root/raw/query.pml -o $root/vs/${filename}_aligned.sdf -S -I -C -D -x 0 -P -N
duration=$(( SECONDS - start ))
echo "Completed in $duration seconds"