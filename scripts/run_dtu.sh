dataset=$1 
workspace=$2 
export CUDA_VISIBLE_DEVICES=$3


python train_dtu.py --dataset DTU -s $dataset --model_path $workspace -r 4 --eval --n_sparse 3 --rand_pcd --iterations 6000 --lambda_dssim 0.6 \
            --densify_grad_threshold 0.001 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.1 \
            --position_lr_init 0.0016 --position_lr_final 0.000016 --position_lr_max_steps 5500 --position_lr_start 500 \
            --test_iterations 100 1000 2000 3000 4500 6000 --save_iterations 100 500 1000 3000 6000\
            --error_tolerance 0.01 \
            --opacity_lr 0.05 --scaling_lr 0.003 \
            --shape_pena 0.005 --opa_pena 0.001 --scale_pena 0.005\

bash ./scripts/copy_mask_dtu.sh

python render.py -s $dataset --model_path $workspace -r 4
python spiral.py -s $dataset --model_path $workspace -r 4

python metrics_dtu.py --model_path $workspace 
