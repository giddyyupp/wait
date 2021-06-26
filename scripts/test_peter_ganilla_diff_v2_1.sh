set -ex
python test.py --dataroot ./datasets/peter --name peter_ganilla_diff_v2_1 --model cycle_gan --netG resnet_fpn --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100
python test.py --dataroot ./datasets/peter --name peter_ganilla_diff_v2_1 --model cycle_gan --netG resnet_fpn --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200
