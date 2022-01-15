import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANWarpModel(BaseModel):
    def name(self):
        return 'CycleGANWarpModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.ordered = not opt.use_warp_speed_ups
        self.rec_bug_fix = opt.rec_bug_fix

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A_1', 'G_A_1', 'cycle_A_1',
                           'idt_A', 'D_B_1', 'G_B', 'cycle_B', 'idt_B_1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A_1', 'real_A_2', 'fake_B_1']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_B = ['real_B', 'fake_A', 'rec_B']
            visual_names_A.append('rec_A_1')
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B_1')

            self.visual_names = visual_names_A + visual_names_B
        else:
            self.visual_names = visual_names_A

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, f"{opt.netG}_warp", opt.norm,
                                        opt.norm_warp, opt.merge_method, opt.final_conv, opt.offset_network_block_cnt,
                                        opt.warp_layer_cnt, not opt.no_dropout, opt.init_type, opt.init_gain,
                                        self.gpu_ids, depth=18, fpn_weights=opt.fpn_weights)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, f"{opt.netG}_warp", opt.norm,
                                        opt.norm_warp, opt.merge_method, opt.final_conv, opt.offset_network_block_cnt,
                                        opt.warp_layer_cnt, not opt.no_dropout, opt.init_type, opt.init_gain,
                                        self.gpu_ids, depth=18, fpn_weights=opt.fpn_weights)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, epoch=0):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_1 = input['A' if AtoB else 'B'][0].to(self.device)
        self.real_A_2 = input['A' if AtoB else 'B'][1].to(self.device)
        # self.real_A_3 = input['A' if AtoB else 'B'][2].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.epoch = epoch

    def forward(self):

        if self.rec_bug_fix:
            self.fake_B_1 = self.netG_A(torch.cat((self.real_A_2, self.real_A_1), 1), ordered=True)
        else:
            self.fake_B_1 = self.netG_A(torch.cat((self.real_A_1, self.real_A_2), 1), ordered=True)

        if self.isTrain:
            if self.ordered:
                self.rec_A_1 = self.netG_B(torch.cat((self.fake_B_1, self.fake_B_1), 1), ordered=self.ordered)
                self.fake_A = self.netG_B(torch.cat((self.real_B, self.real_B), 1), ordered=self.ordered)
                self.rec_B = self.netG_A(torch.cat((self.fake_A, self.fake_A), 1), ordered=self.ordered)
            else:
                self.rec_A_1 = self.netG_B(self.fake_B_1, ordered=self.ordered)
                self.fake_A = self.netG_B(self.real_B, ordered=self.ordered)
                self.rec_B = self.netG_A(self.fake_A, ordered=self.ordered)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B_1 = self.fake_B_pool.query(self.fake_B_1)
        self.loss_D_A_1 = self.backward_D_basic(self.netD_A, self.real_B, fake_B_1)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B_1 = self.backward_D_basic(self.netD_B, self.real_A_1, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            if self.ordered:
                self.idt_A = self.netG_A(torch.cat((self.real_B, self.real_B), 1), ordered=self.ordered)
                self.idt_B_1 = self.netG_B(torch.cat((self.real_A_1, self.real_A_1), 1), ordered=self.ordered)
            else:
                self.idt_A = self.netG_A(self.real_B, ordered=self.ordered)
                self.idt_B_1 = self.netG_B(self.real_A_1, ordered=self.ordered)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.loss_idt_B_1 = self.criterionIdt(self.idt_B_1, self.real_A_1) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B_1, self.loss_idt_B_2, self.loss_idt_B_3 = 0, 0, 0

        # GAN loss D_A(G_A(A_1))
        self.loss_G_A_1 = self.criterionGAN(self.netD_A(self.fake_B_1), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss A_1
        self.loss_cycle_A_1 = self.criterionCycle(self.rec_A_1, self.real_A_1) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss
        self.loss_G = self.loss_G_A_1 + self.loss_G_B + \
                      self.loss_cycle_A_1 + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B_1

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
