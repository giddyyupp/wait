from PIL import Image
import numpy as np
import statistics
import argparse
import os
import re

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--source_dir', type=str)
parser.add_argument('-f', '--fake_dir', type=str)

opt = parser.parse_args()
diff_scores = []

source_files = os.listdir(opt.source_dir)
source_files.sort(key=lambda f: int(re.sub('\D', '', f)))
#print(source_files)
fake_files = os.listdir(opt.fake_dir)
fake_files.sort(key=lambda f: int(re.sub('\D', '', f)))
#print(fake_files)

for i in range(len(source_files) - 1):
    #print(i)
    src_t_1 = Image.open(opt.source_dir + '/' + source_files[i])
    src_t_2 = Image.open(opt.source_dir + '/' + source_files[i + 1])
    
    fake_t_1 = Image.open(opt.fake_dir + '/' + fake_files[i])
    fake_t_2 = Image.open(opt.fake_dir + '/' + fake_files[i + 1])

    src_t_1.load()
    src_t_2.load()
    fake_t_1.load()
    fake_t_2.load()

    src_mat_t_1 = np.asarray(src_t_1, dtype="int32")
    src_mat_t_2 = np.asarray(src_t_2, dtype="int32")
    fake_mat_t_1 = np.asarray(fake_t_1, dtype="int32")
    fake_mat_t_2 = np.asarray(fake_t_2, dtype="int32")

    src_diff = src_mat_t_2 - src_mat_t_1
    fake_diff = fake_mat_t_2 - fake_mat_t_1

    diff = src_diff - fake_diff
    diff_l2 = np.linalg.norm(diff)

    diff_scores.append(diff_l2)

print('Average difference score: %.2f' % statistics.mean(diff_scores))
