#!/bin/bash

#cdpkit='/data/shared/software/CDPKit-head-RH9/Bin'
#project='/home/drose/git/PharmacoMatch'
#data="/data/sharedXL/projects/PharmacoMatch/DUD-E"
cdpkit='<your_path>/CDPKit/Bin'
project='<your_path>/PharmacoMatch'
data="$project/data/DUD-E"
filenames=('actives' 'inactives')

targets=('ACES' 'ADA' 'ANDR' 'EGFR' 'FA10' 'KIT' 'PLK1' 'SRC' 'THRB' 'UROK')

for target in ${targets[@]};
do
    root="$data/$target"
    filename='actives'
    $cdpkit/psdinfo -i $root/raw/$filename.psd -C -P -F
    script -q -c "$cdpkit/psdscreen -d $root/raw/$filename.psd -q $root/raw/query.pml -o $root/vs/${filename}_aligned.sdf -x 0 -t 128 -m BEST-MATCH -a 0 -b 1 -p 0" processing_time.txt
    duration_actives=$(grep "Processing Time:" processing_time.txt | awk '{print $3}') 

    filename='inactives'
    $cdpkit/psdinfo -i $root/raw/$filename.psd -C -P -F
    script -q -c "$cdpkit/psdscreen -d $root/raw/$filename.psd -q $root/raw/query.pml -o $root/vs/${filename}_aligned.sdf -x 0 -t 128 -m BEST-MATCH -a 0 -b 1 -p 0" processing_time.txt
    duration_inactives=$(grep "Processing Time:" processing_time.txt | awk '{print $3}') 
    echo "$target, $duration_actives, $duration_inactives" >> processing_time.csv
done