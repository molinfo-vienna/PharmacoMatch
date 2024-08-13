#!/bin/bash

data='/data/shared/projects/PhectorDB/DUD-E'
filenames=('actives' 'inactives')
#cdpkit='/data/shared/software/CDPKit-head-RH9/Bin'
cdpkit='/data/shared/exchange/tseidel/CDPKit-4DR/Bin'

targets=('ACES' 'ADA' 'ANDR' 'EGFR' 'FA10' 'KIT' 'PLK1' 'SRC' 'THRB' 'UROK')

for target in ${targets[@]};
do
    root="$data/$target"

    filename='actives'
    $cdpkit/psdinfo -i $root/raw/$filename.psd -C -P -F
    # Screening Database Usage
    script -q -c "$cdpkit/psdscreen -d $root/raw/$filename.psd -q $root/raw/query.pml -o $root/vs/${filename}_aligned.sdf -x 0 -t 128 -m BEST-MATCH -a 0 -b 1 -p 0" processing_time.txt
    duration_actives=$(grep "Processing Time:" processing_time.txt | awk '{print $3}') 

    filename='inactives'
    $cdpkit/psdinfo -i $root/raw/$filename.psd -C -P -F
    script -q -c "$cdpkit/psdscreen -d $root/raw/$filename.psd -q $root/raw/query.pml -o $root/vs/${filename}_aligned.sdf -x 0 -t 128 -m BEST-MATCH -a 0 -b 1 -p 0" processing_time.txt
    duration_inactives=$(grep "Processing Time:" processing_time.txt | awk '{print $3}') 
    echo "$target, $duration_actives, $duration_inactives" >> processing_time.csv
done