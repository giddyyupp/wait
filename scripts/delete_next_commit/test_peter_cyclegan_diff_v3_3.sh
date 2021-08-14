set -ex
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 50
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 80
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 100
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 120
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 150
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 180
python test.py --dataroot ./datasets/peter --name peter_cyclegan_diff_v3_3 --model cycle_gan --netG resnet_9blocks --centerCropSize 800 --resize_or_crop center_crop --no_flip --phase test --epoch 200
