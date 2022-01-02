import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_B, get_transform_flow
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import random
import numpy as np


class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        self.transform_B = get_transform_B(opt)

        if opt.optical_flow:
            self.A_size = len(self.A_paths) - 1
            self.flow_dir = os.path.join(opt.dataroot, 'flow')
            self.Flow_paths = make_dataset(self.flow_dir)
            self.Flow_paths = sorted(self.Flow_paths)
            self.transform_flow = get_transform_flow(opt)
            self.Flow_size = len(self.Flow_paths)
            print(self.A_paths)
            print(self.B_paths)
            print(self.Flow_paths)
            print(self.A_size)
            print(self.B_size)
            print(self.Flow_size)

    def __getitem__(self, index):
        # get paths
        A_1_path = self.A_paths[index % self.A_size]

        if self.opt.optical_flow:
            random_index = index + 1
        else:
            random_index = random.randint(index - self.opt.time_gap, index + self.opt.time_gap)

        print(f"1st index: {index}, 2nd index:{random_index}")

        if random_index < 0 or random_index >= self.A_size:
            A_2_path = self.A_paths[index % self.A_size]
        else:
            A_2_path = self.A_paths[random_index % self.A_size]

        if self.opt.optical_flow:
            Flow_path = self.Flow_paths[index % self.A_size]

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # open images
        A1_img = Image.open(A_1_path).convert('RGB')
        A2_img = Image.open(A_2_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.opt.optical_flow:
            Flow_img = np.load(Flow_path)
            # Flow_img = Image.fromarray(Flow_img)
            print(Flow_img.shape)

        if not self.opt.no_flip and random.random() < 0.5:
            A2_img = A2_img.transpose(Image.FLIP_LEFT_RIGHT)
            A1_img = A1_img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.opt.optical_flow:
                Flow_img = np.flip(Flow_img, 1)  # Flow_img.transpose(Image.FLIP_LEFT_RIGHT)  #
                print("Flip")
                print(Flow_img.shape)

        if not self.opt.no_flip and random.random() < 0.5:
            B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)

        # transform images
        A_1 = self.transform(A1_img)
        A_2 = self.transform(A2_img)
        B = self.transform_B(B_img)
        if self.opt.optical_flow:
            Flow = self.transform_flow(Flow_img.copy())
        else:
            Flow = A_1 - A_2  # if optical flow is not selected then use diff image

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A_1[0, ...] * 0.299 + A_1[1, ...] * 0.587 + A_1[2, ...] * 0.114
            A_1 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': [A_1, A_2, Flow], 'B': B,
                'A_paths': [A_1_path, A_2_path], 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
