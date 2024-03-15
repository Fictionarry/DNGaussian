
benchmark=LLFF # LLFF
root_path=../data/distorted/nerf_llff_data

python get_depth_map_for_llff_dtu.py --root_path $root_path --benchmark $benchmark
