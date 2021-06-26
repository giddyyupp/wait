import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
PI = np.pi

class Hough(nn.Module):

    def __init__(self, angle=90, R2_list=[4, 64, 256, 1024], num_classes=80,
                 region_num_visual=9, region_num_temporal=4, vote_field_size=17,
                 voting_map_size_w=128, voting_map_size_h=128,):
        super(Hough, self).__init__()
        self.angle = angle
        self.R2_list = R2_list
        self.region_num_visual = region_num_visual
        self.region_num_temporal = region_num_temporal
        self.num_classes = num_classes
        self.voting_map_size_w = voting_map_size_w
        self.voting_map_size_h = voting_map_size_h
        self.vote_field_size = vote_field_size
        self.deconv_filter_padding = int(self.vote_field_size / 2)
        self.deconv_filters = self._prepare_combined_deconv_filters()

    def  _prepare_combined_deconv_filters(self):

        half_w = int(self.voting_map_size_w / 2)
        half_h = int(self.voting_map_size_h / 2)

        vote_center = torch.tensor([half_h, half_w]).cuda()

        '''visual vote maps'''
        logmap_onehot = self.calculate_logmap((self.voting_map_size_h, self.voting_map_size_w), vote_center,
                                              max_region=17, num_regions = self.region_num_visual)
        weights = logmap_onehot / \
                        torch.clamp(torch.sum(torch.sum(logmap_onehot, dim=0), dim=0).float(), min=1.0)

        start_x = half_h - int(self.vote_field_size/2)
        stop_x  = half_h + int(self.vote_field_size/2) + 1

        start_y = half_w - int(self.vote_field_size/2)
        stop_y  = half_w + int(self.vote_field_size/2) + 1

        visual_filters = weights[start_x:stop_x, start_y:stop_y,:].permute(2,0,1).view(self.region_num_visual, 1,
                                                                     self.vote_field_size, self.vote_field_size)

        '''temporal vote maps'''
        logmap_onehot = self.calculate_logmap((self.voting_map_size_h, self.voting_map_size_w), vote_center,
                                       max_region=8, num_regions = self.region_num_temporal,
                                       R2_list=[0, 64], not_center_region=True)
        weights = logmap_onehot / \
                        torch.clamp(torch.sum(torch.sum(logmap_onehot, dim=0), dim=0).float(), min=1.0)

        temporal_filters = weights[start_x:stop_x, start_y:stop_y,:].permute(2,0,1).view(self.region_num_temporal, 1,
                                                                     self.vote_field_size, self.vote_field_size)

        combined_filters = torch.cat([visual_filters, temporal_filters], dim=0)

        W = nn.Parameter(combined_filters.repeat(self.num_classes, 1, 1, 1))
        W.requires_grad = False

        layers = []
        deconv_kernel = nn.ConvTranspose2d(
            in_channels= (self.region_num_visual + self.region_num_temporal)*self.num_classes,
            out_channels=1*self.num_classes,
            kernel_size=self.vote_field_size,
            padding=self.deconv_filter_padding,
            groups=self.num_classes,
            bias=False)

        with torch.no_grad():
            deconv_kernel.weight = W

        layers.append(deconv_kernel)

        return nn.Sequential(*layers)

    def generate_grid(self, h, w):
        x = torch.arange(0, w).float().cuda()
        y = torch.arange(0, h).float().cuda()
        grid = torch.stack([x.repeat(h), y.repeat(w, 1).t().contiguous().view(-1)], 1)
        return grid.repeat(1, 1).view(-1, 2)

    def calculate_logmap(self, im_size, center, max_region=17, num_regions=17, angle=90,
                         angle_shift=0, R2_list=[4, 64, 256, 1024], not_center_region=False):
        points = self.generate_grid(im_size[0], im_size[1])  # [x,y]
        total_angles = 360 / angle

        # check inside which circle
        y_dif = points[:, 1].cuda() - center[0].float()
        x_dif = points[:, 0].cuda() - center[1].float()

        xdif_2 = x_dif * x_dif
        ydif_2 = y_dif * y_dif
        sum_of_squares = xdif_2 + ydif_2

        # find angle
        arc_angle = (torch.atan2(y_dif, x_dif) * 180 / PI).long()
        arc_angle += angle_shift
        arc_angle[arc_angle < 0] += 360

        angle_id = (arc_angle / angle).long() + 1

        c_region = torch.ones(xdif_2.shape, dtype=torch.long).cuda() * len(R2_list)

        for i in range(len(R2_list) - 1, -1, -1):
            region = R2_list[i]
            c_region[(sum_of_squares) <= region] = i

        results = angle_id + (c_region - 1) * total_angles
        results[results < 0] = 0

        logmap = results.view(im_size[0], im_size[1])

        if not_center_region:
            logmap[(logmap == 0)] = 1
            logmap = logmap - 1

        logmap_onehot = torch.nn.functional.one_hot(logmap.long(), num_classes=max_region).float()
        logmap_onehot = logmap_onehot[:, :, :num_regions]

        return logmap_onehot

    def forward(self, visual_voting_map, temporal_voting_maps, targets=None):

        batch_size, channels, width, height = visual_voting_map.shape
        visual_map = visual_voting_map.view(batch_size, self.region_num_visual, self.num_classes, width, height)

        temporal_maps = []
        for temp_voting_map in temporal_voting_maps:
            temporal_map_i = temp_voting_map.view(batch_size, self.region_num_temporal, self.num_classes, width, height)
            temporal_maps.append(torch.unsqueeze(temporal_map_i, 1))

        temporal_map = torch.sum(torch.cat(temporal_maps, dim=1), 1)
        # torch.max(torch.cat(temporal_maps, dim=1), 1)[0]
        # torch.mean(torch.cat(temporal_maps, dim=1), dim=1)

        voting_map = torch.cat([visual_map, temporal_map], dim=1)

        voting_map = voting_map.permute(0, 2, 1, 3, 4)
        voting_map = voting_map.reshape(batch_size, -1, width, height)
        heatmap = self.deconv_filters(voting_map)

        return heatmap