set -ex
python test.py --dataroot ./datasets/viper/test --name viper_cyclegan_diff_v3_1 --model cycle_gan --netG resnet_9blocks --no_flip --phase test --epoch 15 --num_test 1500
python test.py --dataroot ./datasets/viper/test --name viper_cyclegan_diff_v3_1 --model cycle_gan --netG resnet_9blocks --no_flip --phase test --epoch 20 --num_test 1500
