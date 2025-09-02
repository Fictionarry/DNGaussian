dataset=$1 
workspace=$2 
export CUDA_VISIBLE_DEVICES=$3

## (sorry for that this part is not so elegent)


## For the later scenes, we do not need to apply soft depth supervision.

## for     materials, drums
if [[ $dataset =~ "drums" ]] || [[ $dataset =~ "materials" ]]; then
    echo "setting 1"
    python train_blender.py -s $dataset --model_path $workspace -r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background \
                --densify_grad_threshold 0.0005 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --hard_depth_start 0 --soft_depth_start 9999999 \
                --split_opacity_thresh 0.1 --error_tolerance 0.001 \
                --scaling_lr 0.005 \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000

    python render.py -s $dataset --model_path $workspace -r 2
    python metrics.py --model_path $workspace 

fi


## for      ship, lego, ficus, hotdog     SH peforms better

if [[ $dataset =~ ship ]] || [[ $dataset =~ lego ]] || [[ $dataset =~ ficus ]] || [[ $dataset =~ hotdog ]]; then
    echo "setting 2"
    
    python train_blender.py -s $dataset --model_path $workspace -r 2 --eval --n_sparse 8 --rand_pcd --iterations 6000 --lambda_dssim 0.2 --white_background \
                --densify_grad_threshold 0.0005 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --hard_depth_start 0 \
                --error_tolerance 0.01 \
                --scaling_lr 0.005 \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
                --use_SH 


    python render_sh.py -s $dataset --model_path $workspace -r 2
    python metrics.py --model_path $workspace 

fi




## for      chair, mic       the sampled views has a fully covering range so the model do not need monocular depth any more....

if [[ $dataset =~ chair ]] || [[ $dataset =~ mic ]]; then
    echo "setting 3"
    python train_blender.py -s $dataset --model_path $workspace -r 2 --eval --n_sparse 8 --rand_pcd --iterations 30000 --lambda_dssim 0.2 --white_background \
                --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 15000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 30000 --position_lr_start 0 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --hard_depth_start 99999 \
                --error_tolerance 0.2 \
                --scaling_lr 0.005 \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
                --use_SH

    python render_sh.py -s $dataset --model_path $workspace -r 2
    python metrics.py --model_path $workspace 
fi
