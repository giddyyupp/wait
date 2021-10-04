import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F
from models.deform_conv.modules.deform_conv import DeformConv
from models.hough_module import Hough

import math
import torch.utils.model_zoo as model_zoo
import sys


BN_MOMENTUM = 0.1

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[], depth=18, fpn_weights=[1.0, 1.0, 1.0, 1.0]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_warp':
        net = ResnetGeneratorWarp(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_hough':
        net = ResnetGeneratorHough(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_fpn':
        # Create the regular GANILLA model
        net = resnet18(input_nc, output_nc, ngf, fpn_weights, use_dropout=use_dropout, pretrained=False)
    elif netG == 'resnet_fpn_warp':
        # Create the GANILLA + feature warp model
        net = resnet18_warp(input_nc, output_nc, ngf, fpn_weights, use_dropout=use_dropout, pretrained=False)
    elif netG == 'resnet_fpn_hough':
        # Create the model
        print("Not implemented yet!")
        # net = resnet18_hough(input_nc, output_nc, ngf, fpn_weights, use_dropout=use_dropout, pretrained=False)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Feature Warping
class ResnetGeneratorWarp(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGeneratorWarp, self).__init__()
        self.block_count = 8
        self.cycle_consistency_finetune = False
        self.warping_reverse = False

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        k = 3
        warp_out = 3  # feature:64, image:3
        inner_ch = 3  # feature:64, image:3

        self.offset_feats = self._compute_chain_of_basic_blocks(warp_out, inner_ch, 1, 1, 2,
                                                                warp_out, self.block_count).cuda()

        #### Offsets
        self.offsets1 = self._single_conv(inner_ch, k, k, 3, warp_out).cuda()
        self.offsets2 = self._single_conv(inner_ch, k, k, 6, warp_out).cuda()
        self.offsets3 = self._single_conv(inner_ch, k, k, 12, warp_out).cuda()
        self.offsets4 = self._single_conv(inner_ch, k, k, 18, warp_out).cuda()
        self.offsets5 = self._single_conv(inner_ch, k, k, 24, warp_out).cuda()

        #### Deformable Conv
        self.deform_conv1 = self._deform_conv(warp_out, k, k, 3, warp_out).cuda()
        self.deform_conv2 = self._deform_conv(warp_out, k, k, 6, warp_out).cuda()
        self.deform_conv3 = self._deform_conv(warp_out, k, k, 12, warp_out).cuda()
        self.deform_conv4 = self._deform_conv(warp_out, k, k, 18, warp_out).cuda()
        self.deform_conv5 = self._deform_conv(warp_out, k, k, 24, warp_out).cuda()

        model_final = [nn.ReflectionPad2d(3)]
        model_final += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_final += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_final = nn.Sequential(*model_final)
        self.tanh = nn.Tanh()

    def _compute_chain_of_basic_blocks(self, nc, ic, kh, kw, dd, dg, b):
        num_blocks = b
        block = BasicBlock
        in_ch = ic
        out_ch = ic
        stride = 1

        ######
        downsample = nn.Sequential(
            nn.Conv2d(
                nc,
                in_ch,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.InstanceNorm2d(in_ch),
            #nn.BatchNorm2d(
            #    in_ch,
            #    momentum=BN_MOMENTUM
            #),
        )

        ##########
        layers = []
        layers.append(
            block(
                nc,
                out_ch,
                stride,
                downsample
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_ch,
                    out_ch
                )
            )

        return nn.Sequential(*layers)

    def _single_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(
            nc,
            dg * 2 * kh * kw,
            kernel_size=(3, 3),
            stride=(1, 1),
            dilation=(dd, dd),
            padding=(1 * dd, 1 * dd),
            bias=False)
        return conv

    def _deform_conv(self, nc, kh, kw, dd, dg):
        conv_offset2d = DeformConv(
            nc,
            nc, (kh, kw),
            stride=1,
            padding=int(kh / 2) * dd,
            dilation=dd,
            deformable_groups=dg)
        return conv_offset2d

    def forward(self, inputs, ordered=True):
        if ordered:
            batch_size = inputs.size(0)
            ref_x = inputs[:, 0:self.input_nc, :, :]
            sup_x = inputs[:, self.input_nc:, :, :]
            x = torch.cat((ref_x, sup_x), 0).contiguous()
        else:  # un-ordered dataset
            x = inputs

        out = self.model(x)
        # below is for warping on 3 channel image
        out = self.model_final(out)

        if not ordered:  # un-ordered dataset
            return out

        """
            Warping phase
        """
        ref_x = out[:batch_size, :, :, :].contiguous()
        sup_x = out[batch_size:, :, :, :].contiguous()

        #### cycle consistency
        if self.cycle_consistency_finetune:
            diff_x_back = ref_x - sup_x
            diff_x_forw = sup_x - ref_x
        else:
            diff_x = ref_x - sup_x
        ###########

        if self.warping_reverse:
            diff_x = sup_x - ref_x
            sup_x = ref_x

        ### cycle consistency
        if self.cycle_consistency_finetune:
            off_feats_cuda = self.offset_feats(diff_x_forw.cuda())
            sup_x_cuda = ref_x.cuda()
        else:
            off_feats_cuda = self.offset_feats(diff_x.cuda())
            sup_x_cuda = sup_x.cuda()

        #########

        off1 = self.offsets1(off_feats_cuda)
        warped_x1 = self.deform_conv1(sup_x_cuda.contiguous(), off1.contiguous())

        off2 = self.offsets2(off_feats_cuda)
        warped_x2 = self.deform_conv2(sup_x_cuda.contiguous(), off2.contiguous())
        
        off3 = self.offsets3(off_feats_cuda)
        warped_x3 = self.deform_conv3(sup_x_cuda.contiguous(), off3.contiguous())
        
        off4 = self.offsets4(off_feats_cuda)
        warped_x4 = self.deform_conv4(sup_x_cuda.contiguous(), off4.contiguous())
        
        off5 = self.offsets5(off_feats_cuda)
        warped_x5 = self.deform_conv5(sup_x_cuda.contiguous(), off5.contiguous())

        x = 0.20 * (warped_x1 + warped_x2 + warped_x3 + warped_x4 + warped_x5)
        #x = warped_x1

        #### backwards
        if self.cycle_consistency_finetune:
            off_feats_cuda = self.offset_feats(diff_x_back.cuda())
            sup_x_cuda = x

            off1 = self.offsets1(off_feats_cuda)
            warped_x1 = self.deform_conv1(sup_x_cuda, off1)

            # off2 = self.offsets2(off_feats_cuda)
            # warped_x2 = self.deform_conv2(sup_x_cuda, off2)
            #
            # off3 = self.offsets3(off_feats_cuda)
            # warped_x3 = self.deform_conv3(sup_x_cuda, off3)
            #
            # off4 = self.offsets4(off_feats_cuda)
            # warped_x4 = self.deform_conv4(sup_x_cuda, off4)
            #
            # off5 = self.offsets5(off_feats_cuda)
            # warped_x5 = self.deform_conv5(sup_x_cuda, off5)

            # x = 0.25 * (warped_x1 + warped_x2 + warped_x3 + warped_x4)
            x = warped_x1

        ###############

        return self.tanh(x)  # self.model_final(x) ## return x for 3 channel warping


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # , momentum=BN_MOMENTUM
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # , momentum=BN_MOMENTUM
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Hough Voting
class ResnetGeneratorHough(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGeneratorHough, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # to enaoble hough voting
        visual_vot_head = [nn.Conv2d(ngf, 27, kernel_size=3, padding=1), nn.ReLU(True)]  # 3*9 = 27
        temporal_vot_head = [nn.Conv2d(ngf, 12, kernel_size=3, padding=1), nn.ReLU(True)]  # 3*4 = 12
        final_layer = [nn.ReflectionPad2d(3), nn.Conv2d(3, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.visual_vot_head = nn.Sequential(*visual_vot_head)
        self.temporal_vot_head = nn.Sequential(*temporal_vot_head)
        self.final_layer = nn.Sequential(*final_layer)

        self.hough_voting = Hough(region_num_visual=9,
                                  region_num_temporal=4,
                                  vote_field_size=17,
                                  num_classes=3)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        ref_inp = inputs[:, 0:self.input_nc, :, :]  # current frame
        sup_inp = inputs[:, self.input_nc:, :, :]  # supporting frame
        x_ = torch.cat((ref_inp, sup_inp), 0)
        x = self.model(x_)

        ref_out = x[:batch_size, :, :, :]
        sup_out = x[batch_size:, :, :, :]

        visual_voting_map = self.visual_vot_head(ref_out)
        temporal_voting_map = [self.temporal_vot_head(ref_out - sup_out)]
        img = self.hough_voting(visual_voting_map, temporal_voting_map)
        return self.final_layer(img)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


######## GANILLA #########


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock_orj(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_orj, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.rp2 = nn.ReflectionPad2d(1)
        self.conv2 = conv3x3(planes, planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.rp1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)

        out = self.rp2(out)
        out = self.conv2(out)
        out = self.in2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_Ganilla(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, use_dropout, stride=1):
        super(BasicBlock_Ganilla, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        self.rp2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.out_planes = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

            self.final_conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.expansion * planes * 2, self.expansion * planes, kernel_size=3, stride=1,
                                        padding=0, bias=False),
                nn.InstanceNorm2d(self.expansion * planes)
            )
        else:
            self.final_conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(planes*2, planes, kernel_size=3, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.rp1(x))))
        if self.use_dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(self.rp2(out)))
        inputt = self.shortcut(x)
        catted = torch.cat((out, inputt), 1)
        out = self.final_conv(catted)
        out = F.relu(out)
        return out


class PyramidFeatures_v3(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(PyramidFeatures_v3, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp1 = nn.ReflectionPad2d(1)
        #self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp2 = nn.ReflectionPad2d(1)
        #self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.rp3 = nn.ReflectionPad2d(1)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        #self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.rp4 = nn.ReflectionPad2d(1)
        # self.P2_2 = nn.Conv2d(feature_size, feature_size/2, kernel_size=3, stride=1, padding=0)

        self.P1_1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp5 = nn.ReflectionPad2d(1)
        self.P1_2 = nn.Conv2d(feature_size, feature_size/2, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)

        P1_x = self.P1_1(P3_upsampled_x)
        P1_upsampled_x = self.P1_upsampled(P1_x)
        P2_x = self.rp5(P1_upsampled_x)
        P2_x = self.P1_2(P2_x)

        return P2_x


class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, fpn_weights, feature_size=128):
        super(PyramidFeatures, self).__init__()

        self.sum_weights = fpn_weights #[1.0, 0.5, 0.5, 0.5]

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp1 = nn.ReflectionPad2d(1)
        #self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp2 = nn.ReflectionPad2d(1)
        #self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp3 = nn.ReflectionPad2d(1)
        #self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp4 = nn.ReflectionPad2d(1)
        self.P2_2 = nn.Conv2d(int(feature_size), int(feature_size/2), kernel_size=3, stride=1, padding=0)

        #self.P1_1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp5 = nn.ReflectionPad2d(1)
        #self.P1_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        i = 0
        P5_x = self.P5_1(C5) * self.sum_weights[i]
        P5_upsampled_x = self.P5_upsampled(P5_x)
        #P5_x = self.rp1(P5_x)
        # #P5_x = self.P5_2(P5_x)
        i += 1
        P4_x = self.P4_1(C4) * self.sum_weights[i]
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        #P4_x = self.rp2(P4_x)
        # #P4_x = self.P4_2(P4_x)
        i += 1
        P3_x = self.P3_1(C3) * self.sum_weights[i]
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        #P3_x = self.rp3(P3_x)
        #P3_x = self.P3_2(P3_x)
        i += 1
        P2_x = self.P2_1(C2) * self.sum_weights[i]
        P2_x = P2_x * self.sum_weights[2] + P3_upsampled_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        P2_x = self.rp4(P2_upsampled_x)
        P2_x = self.P2_2(P2_x)

        return P2_x


class ResNet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, fpn_weights, block, layers, use_dropout):
        self.inplanes = ngf
        super(ResNet, self).__init__()

        # first conv
        self.pad1 = nn.ReflectionPad2d(input_nc)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Output layer
        self.pad3 = nn.ReflectionPad2d(output_nc)
        self.conv2 = nn.Conv2d(64, output_nc, 7)
        self.tanh = nn.Tanh()

        if block == BasicBlock_orj:
            # residuals
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        elif block == BasicBlock_Ganilla:
            # residuals
            self.layer1 = self._make_layer_ganilla(block, 64, layers[0], use_dropout, stride=1)
            self.layer2 = self._make_layer_ganilla(block, 128, layers[1], use_dropout, stride=2)
            self.layer3 = self._make_layer_ganilla(block, 128, layers[2], use_dropout, stride=2)
            self.layer4 = self._make_layer_ganilla(block, 256, layers[3], use_dropout, stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        else:
            print("Block Type is not Correct")
            sys.exit()

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], fpn_weights)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

        # self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_ganilla(self, block, planes, blocks, use_dropout, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, use_dropout, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        img_batch = inputs

        x = self.pad1(img_batch)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.fpn([x1, x2, x3, x4]) # use all resnet layers

        out = self.pad3(out)
        out = self.conv2(out)
        out = self.tanh(out)

        return out


class ResNetWarp(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, fpn_weights, block, layers, use_dropout):
        self.inplanes = ngf
        super(ResNetWarp, self).__init__()

        self.block_count = 8
        self.cycle_consistency_finetune = False
        self.warping_reverse = False

        self.input_nc = input_nc
        self.output_nc = output_nc

        # first conv
        self.pad1 = nn.ReflectionPad2d(input_nc)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Output layer
        self.pad3 = nn.ReflectionPad2d(output_nc)
        self.conv2 = nn.Conv2d(ngf, output_nc, 7, padding=0)
        self.tanh = nn.Tanh()

        if block == BasicBlock_orj:
            # residuals
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        elif block == BasicBlock_Ganilla:
            # residuals
            self.layer1 = self._make_layer_ganilla(block, 64, layers[0], use_dropout, stride=1)
            self.layer2 = self._make_layer_ganilla(block, 128, layers[1], use_dropout, stride=2)
            self.layer3 = self._make_layer_ganilla(block, 128, layers[2], use_dropout, stride=2)
            self.layer4 = self._make_layer_ganilla(block, 256, layers[3], use_dropout, stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        else:
            print("Block Type is not Correct")
            sys.exit()

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], fpn_weights)

        k = 3
        warp_out = 3  # feature:64, image:3
        inner_ch = 3  # feature:64, image:3

        self.offset_feats = self._compute_chain_of_basic_blocks(warp_out, inner_ch, 1, 1, 2,
                                                                warp_out, self.block_count).cuda()

        #### Offsets
        self.offsets1 = self._single_conv(inner_ch, k, k, 3, warp_out).cuda()
        self.offsets2 = self._single_conv(inner_ch, k, k, 6, warp_out).cuda()
        self.offsets3 = self._single_conv(inner_ch, k, k, 12, warp_out).cuda()
        self.offsets4 = self._single_conv(inner_ch, k, k, 18, warp_out).cuda()
        self.offsets5 = self._single_conv(inner_ch, k, k, 24, warp_out).cuda()

        #### Deformable Conv
        self.deform_conv1 = self._deform_conv(warp_out, k, k, 3, warp_out).cuda()
        self.deform_conv2 = self._deform_conv(warp_out, k, k, 6, warp_out).cuda()
        self.deform_conv3 = self._deform_conv(warp_out, k, k, 12, warp_out).cuda()
        self.deform_conv4 = self._deform_conv(warp_out, k, k, 18, warp_out).cuda()
        self.deform_conv5 = self._deform_conv(warp_out, k, k, 24, warp_out).cuda()

        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

        # self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_ganilla(self, block, planes, blocks, use_dropout, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, use_dropout, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _compute_chain_of_basic_blocks(self, nc, ic, kh, kw, dd, dg, b):
        num_blocks = b
        block = BasicBlock
        in_ch = ic
        out_ch = ic
        stride = 1

        ######
        downsample = nn.Sequential(
            nn.Conv2d(
                nc,
                in_ch,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.InstanceNorm2d(in_ch),
            # nn.BatchNorm2d(
            #     in_ch,
            #     momentum=BN_MOMENTUM
            # ),
        )

        ##########
        layers = []
        layers.append(
            block(
                nc,
                out_ch,
                stride,
                downsample
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_ch,
                    out_ch
                )
            )

        return nn.Sequential(*layers)

    def _single_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(
            nc,
            dg * 2 * kh * kw,
            kernel_size=(3, 3),
            stride=(1, 1),
            dilation=(dd, dd),
            padding=(1 * dd, 1 * dd),
            bias=False)
        return conv

    def _deform_conv(self, nc, kh, kw, dd, dg):
        conv_offset2d = DeformConv(
            nc,
            nc, (kh, kw),
            stride=1,
            padding=int(kh / 2) * dd,
            dilation=dd,
            deformable_groups=dg)
        return conv_offset2d

    def forward(self, inputs):

        batch_size = inputs.size(0)
        ref_x = inputs[:, 0:self.input_nc, :, :]
        sup_x = inputs[:, self.input_nc:, :, :]
        x = torch.cat((ref_x, sup_x), 0)

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.fpn([x1, x2, x3, x4])

        # moved here for warping on 3 color image
        out = self.pad3(out)
        out = self.conv2(out)
        out = self.tanh(out)

        """
            Warping phase
        """
        ref_x = out[:batch_size, :, :, :].contiguous()
        sup_x = out[batch_size:, :, :, :].contiguous()

        #### cycle consistency
        if self.cycle_consistency_finetune:
            diff_x_back = ref_x - sup_x
            diff_x_forw = sup_x - ref_x
        else:
            diff_x = ref_x - sup_x
        ###########

        if self.warping_reverse:
            diff_x = sup_x - ref_x
            sup_x = ref_x

        ### cycle consistency
        if self.cycle_consistency_finetune:
            off_feats_cuda = self.offset_feats(diff_x_forw.cuda())
            sup_x_cuda = ref_x.cuda()
        else:
            off_feats_cuda = self.offset_feats(diff_x.cuda())
            sup_x_cuda = sup_x.cuda()

        #########

        off1 = self.offsets1(off_feats_cuda)
        warped_x1 = self.deform_conv1(sup_x_cuda, off1)

        off2 = self.offsets2(off_feats_cuda)
        warped_x2 = self.deform_conv2(sup_x_cuda, off2)

        off3 = self.offsets3(off_feats_cuda)
        warped_x3 = self.deform_conv3(sup_x_cuda, off3)

        off4 = self.offsets4(off_feats_cuda)
        warped_x4 = self.deform_conv4(sup_x_cuda, off4)

        off5 = self.offsets5(off_feats_cuda)
        warped_x5 = self.deform_conv5(sup_x_cuda, off5)

        x = 0.2 * (warped_x1 + warped_x2 + warped_x3 + warped_x4 + warped_x5)
        # x = warped_x1

        #### backwards
        if self.cycle_consistency_finetune:
            off_feats_cuda = self.offset_feats(diff_x_back.cuda())
            sup_x_cuda = x

            off1 = self.offsets1(off_feats_cuda)
            warped_x1 = self.deform_conv1(sup_x_cuda, off1)

            # off2 = self.offsets2(off_feats_cuda)
            # warped_x2 = self.deform_conv2(sup_x_cuda, off2)
            #
            # off3 = self.offsets3(off_feats_cuda)
            # warped_x3 = self.deform_conv3(sup_x_cuda, off3)
            #
            # off4 = self.offsets4(off_feats_cuda)
            # warped_x4 = self.deform_conv4(sup_x_cuda, off4)
            #
            # off5 = self.offsets5(off_feats_cuda)
            # warped_x5 = self.deform_conv5(sup_x_cuda, off5)

            # x = 0.2 * (warped_x1 + warped_x2 + warped_x3 + warped_x4 + warped_x5)
            x = warped_x1

        ###############

        # # final conv to generate 3 channel color image.
        # out = self.pad3(x)
        # out = self.conv2(out)
        # out = self.tanh(out)

        return self.tanh(x)  # out ## return x for 3 color image warping else out


def resnet18(input_nc, output_nc, ngf, fpn_weights, use_dropout, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(input_nc, output_nc, ngf, fpn_weights, BasicBlock_Ganilla, [2, 2, 2, 2], use_dropout,  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet18_warp(input_nc, output_nc, ngf, fpn_weights, use_dropout, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetWarp(input_nc, output_nc, ngf, fpn_weights, BasicBlock_Ganilla, [2, 2, 2, 2], use_dropout,  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model
