dataset_dir=$1
exp_id=$2
gen_type=$3

set -ex
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 180
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50
