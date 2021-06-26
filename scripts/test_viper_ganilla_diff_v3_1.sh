set -ex
python test.py --dataroot ./datasets/viper/test --name viper_ganilla_diff_v3_1 --model cycle_gan --netG resnet_fpn --no_flip --phase test --epoch 5 --num_test 1500

