import os
import sys
import shutil


def getint(name):
    basename = os.path.basename(name)
    alpha = basename.split('_')
    return int(alpha[1])

# convert source images
#data_dir = '../data/test/input/flowers_dandelion/testA'
#target_dir = '../data/test/input/flowers_dandelion/video1'

# convert result images
task_id = sys.argv[1]
dataset_id = sys.argv[2]
data_dir = f'data/test/{task_id}/fwe/{dataset_id}/fake'
target_dir = f'data/test/{task_id}/fwe/{dataset_id}/video1'

png_files = os.listdir(data_dir)
png_files.sort(key=getint)
os.makedirs(target_dir, exist_ok=True)

for ii, ff in enumerate(png_files):
    #print(ff)
    new_name = '%05d.png' % ii
    shutil.copy(os.path.join(data_dir, ff), os.path.join(target_dir, new_name))
