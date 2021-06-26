set -ex
python test.py --dataroot ./datasets/axel --name axel_ganilla_diff_v3_1 --model cycle_gan --netG resnet_fpn --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 30
python test.py --dataroot ./datasets/axel --name axel_ganilla_diff_v3_1 --model cycle_gan --netG resnet_fpn --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 40
