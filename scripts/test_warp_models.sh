dataset_dir=$1
exp_id=$2
gen_type=$3

set -ex
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 140 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 130 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120 --time_gap 0
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 110 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 90 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50 --time_gap 0

# peters
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 250 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 240 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 220 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 180 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80 --time_gap 0
#python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50 --time_gap 0
