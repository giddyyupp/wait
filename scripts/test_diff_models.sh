dataset_dir=$1
exp_id=$2
gen_type=$3
dataset=$4

set -ex

if [ "$dataset" == "axel" ]
then

echo "Test Model: Axel"
# axel
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 140
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 130
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 110
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 90
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50

else

echo "Test Model: Peter"

# peters
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 300
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 290
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 270
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 250
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 240
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 220
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 180
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50

fi
