dataset_dir=$1
exp_id=$2
gen_type=$3
dataset=$4
flag_norm=$5
param_norm=$6
bug_fix=$7
speed_up=$8

set -ex

if [ "$dataset" == "axel" ]
then

echo "Test Model: Axel"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 190 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 180 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 170 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 160 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 140 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 130 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 110 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 90 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "sum"

else

echo "Test Model: Peter"
# peters
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 300 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 290 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 270 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 250 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 240 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 220 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 180 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"
python test.py --dataroot ${dataset_dir} --name ${exp_id} --model cycle_gan_warp --netG ${gen_type} --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50 --time_gap 0 ${flag_norm} ${param_norm} ${bug_fix} ${speed_up} --final_conv --merge_method "concat"

fi
