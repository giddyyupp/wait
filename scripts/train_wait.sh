set -ex
python train.py --dataroot ./datasets/bp_dataset --name bp_wait --model cycle_gan_warp --netG resnet_9blocks --centerCropSize 256 --resize_or_crop resize_and_centercrop --batch_size 8 --lr 0.0008 --niter_decay 200 --verbose --norm_warp "batch" --use_warp_speed_ups --rec_bug_fix --final_conv --merge_method "concat" --time_gap 5 --offset_network_block_cnt 10 --warp_layer_cnt 5
