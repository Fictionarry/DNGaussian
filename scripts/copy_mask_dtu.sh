base="output/dtu/"
mask_path="data/dtu/submission_data/idrmasks"

for scan_id in scan30 scan34 scan41 scan45  scan82 scan103  scan38  scan21 scan40  scan55  scan63  scan31  scan8  scan110  scan114
do  
    if [ -d $base/$scan_id ]; then
        # rm -r $base/$scan_id/mask
        mkdir $base/$scan_id/mask
        id=0
        if [ -d ${mask_path}/$scan_id/mask ]; then
            for file in ${mask_path}/scan8/*
            do  
                # echo $file
                file_name=$(printf "%05d" $id).png;
                cp ${file//scan8/$scan_id'/mask'} $base/$scan_id/mask/$file_name
                ((id = id + 1))
            done

            else

            for file in ${mask_path}/$scan_id/*
            do
                # echo $file
                file_name=$(printf "%05d" $id).png;
                cp $file $base/$scan_id/mask/$file_name
                ((id = id + 1))
            done
        fi
    fi
    
done