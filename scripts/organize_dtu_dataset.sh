rectified_path=$1 # ../data/DTU/Rectified


for scan_id in scan30 scan34 scan41 scan45  scan82 scan103  scan38  scan21 scan40  scan55  scan63  scan31  scan8  scan110  scan114
do  
    echo $scan_id
    mkdir -p ./data/dtu/$scan_id/input
    cp $rectified_path/$scan_id/*_3_r5000.png ./data/dtu/$scan_id/input/
done