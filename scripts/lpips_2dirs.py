import argparse
import os
import lpips
import statistics

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s','--source_dir', type=str, default='./imgs/ex_dir0')
parser.add_argument('-f','--fake_dir', type=str, default='./imgs/ex_dir1')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='vgg',version=opt.version)
if(opt.use_gpu):
    loss_fn.cuda()

# crawl directories
source_files = sorted(os.listdir(opt.source_dir))
fake_files = sorted(os.listdir(opt.fake_dir))
lpips_scores = []

for source_file, fake_file in zip(source_files, fake_files):
    # Load images
    source_img = lpips.im2tensor(lpips.load_image(os.path.join(opt.source_dir, source_file))) # RGB image from [-1,1]
    fake_img = lpips.im2tensor(lpips.load_image(os.path.join(opt.fake_dir, fake_file)))

    if(opt.use_gpu):
        source_img = source_img.cuda()
        fake_img = fake_img.cuda()

    # Compute distance
    dist01 = loss_fn.forward(source_img, fake_img)
    print('%s %s: %.3f' % (source_file, fake_file, dist01.item()))
    lpips_scores.append(dist01.item())

print('Avg LPIPS score: %.14f' % statistics.mean(lpips_scores))
