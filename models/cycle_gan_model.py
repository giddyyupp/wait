import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=20.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_fake_diff', type=float, default=1.0, help='weight for fake diff loss')
            parser.add_argument('--lambda_real_fake_diff', type=float, default=1.0, help='weight for real fake diff loss')
            parser.add_argument('--lambda_fake_rec_diff', type=float, default=1.0, help='weight for fake rec diff loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['D_A_1', 'D_A_2', 'D_A_3', 'G_A_1', 'G_A_2', 'G_A_3', 'cycle_A_1', 'cycle_A_2', 'cycle_A_3',
        #                    'idt_A', 'D_B_1', 'D_B_2', 'D_B_3', 'G_B', 'cycle_B', 'idt_B_1', 'idt_B_2', 'idt_B_3']
        self.loss_names = ['D_A_1', 'D_A_2', 'G_A_1', 'G_A_2', 'cycle_A_1', 'cycle_A_2',
                           'idt_A', 'D_B_1', 'D_B_2', 'G_B', 'cycle_B', 'idt_B_1', 'idt_B_2'] # , 'idt_B_3'
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # visual_names_A = ['real_A_1', 'real_A_2', 'real_A_3', 'fake_B_1', 'fake_B_2', 'fake_B_3',
        #                   'rec_A_1', 'rec_A_2', 'rec_A_3']
        visual_names_A = ['real_A_1', 'real_A_2', 'fake_B_1', 'fake_B_2', 'rec_A_1', 'rec_A_2']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B_1')
            visual_names_B.append('idt_B_2')
            # visual_names_B.append('idt_B_3')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, None,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, depth=18,
                                        fpn_weights=opt.fpn_weights)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, None,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, depth=18,
                                        fpn_weights=opt.fpn_weights)

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
            self.criterionFakeDiff = torch.nn.L1Loss()
            self.criterionRealFakeDiff = torch.nn.L1Loss()
            self.criterionFakeRecDiff = torch.nn.L1Loss()
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
        self.real_A_3 = input['A' if AtoB else 'B'][2].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.epoch = epoch

    def forward(self):
        self.fake_B_1 = self.netG_A(self.real_A_1)     # I_t
        self.fake_B_2 = self.netG_A(self.real_A_2)     # I_t+k

        self.rec_A_1 = self.netG_B(self.fake_B_1)
        self.rec_A_2 = self.netG_B(self.fake_B_2)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        if self.epoch >= 10:
            # self.set_requires_grad([self.netG_A, self.netG_B], False)
            self.fake_B_3 = self.netG_A(self.real_A_3)     # D_I
            self.rec_A_3 = self.netG_B(self.fake_B_3)
            # self.set_requires_grad([self.netG_A, self.netG_B], True)


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

        fake_B_2 = self.fake_B_pool.query(self.fake_B_2)
        self.loss_D_A_2 = self.backward_D_basic(self.netD_A, self.real_B, fake_B_2)

        # fake_B_3 = self.fake_B_pool.query(self.fake_B_3)
        # self.loss_D_A_3 = self.backward_D_basic(self.netD_A, self.real_B, fake_B_3)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B_1 = self.backward_D_basic(self.netD_B, self.real_A_1, fake_A)
        self.loss_D_B_2 = self.backward_D_basic(self.netD_B, self.real_A_2, fake_A)
        # self.loss_D_B_3 = self.backward_D_basic(self.netD_B, self.real_A_3, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A

        if self.epoch >= 10:
            lambda_B = 30.0
        else:
            lambda_B = self.opt.lambda_B

        lambda_fake_diff = self.opt.lambda_fake_diff
        lambda_real_fake_diff = self.opt.lambda_real_fake_diff
        lambda_fake_rec_diff = self.opt.lambda_fake_rec_diff

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B_1 = self.netG_B(self.real_A_1)
            self.loss_idt_B_1 = self.criterionIdt(self.idt_B_1, self.real_A_1) * lambda_A * lambda_idt

            self.idt_B_2 = self.netG_B(self.real_A_2)
            self.loss_idt_B_2 = self.criterionIdt(self.idt_B_2, self.real_A_2) * lambda_A * lambda_idt

            # self.idt_B_3 = self.netG_B(self.real_A_3)
            # self.loss_idt_B_3 = self.criterionIdt(self.idt_B_3, self.real_A_3) * lambda_A * lambda_idt   # will be deleted
        else:
            self.loss_idt_A = 0
            self.loss_idt_B_1, self.loss_idt_B_2, self.loss_idt_B_3 = 0, 0, 0

        # GAN loss D_A(G_A(A_1))
        self.loss_G_A_1 = self.criterionGAN(self.netD_A(self.fake_B_1), True)
        # GAN loss D_A(G_A(A_2))
        self.loss_G_A_2 = self.criterionGAN(self.netD_A(self.fake_B_2), True)
        # GAN loss D_A(G_A(A_3))
        # self.loss_G_A_3 = self.criterionGAN(self.netD_A(self.fake_B_3), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss A_1
        self.loss_cycle_A_1 = self.criterionCycle(self.rec_A_1, self.real_A_1) * lambda_A
        # Forward cycle loss A_2
        self.loss_cycle_A_2 = self.criterionCycle(self.rec_A_2, self.real_A_2) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss
        # self.loss_G = self.loss_G_A_1 + self.loss_G_A_2 + self.loss_G_B + \
        #               self.loss_cycle_A_1 + self.loss_cycle_A_2 + self.loss_cycle_A_3 + self.loss_cycle_B + \
        #               self.loss_diff + self.loss_idt_A + self.loss_idt_B_1 + self.loss_idt_B_2 + self.loss_idt_B_3
        self.loss_G = self.loss_G_A_1 + self.loss_G_A_2 + self.loss_G_B + \
                      self.loss_cycle_A_1 + self.loss_cycle_A_2 + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B_1 + self.loss_idt_B_2 #+ self.loss_idt_B_3

        if self.epoch >= 10:
            # Forward cycle loss A_3
            self.loss_cycle_A_3 = self.criterionCycle(self.rec_A_3, self.real_A_3) * lambda_A  # will be deleted
            self.loss_G += self.loss_cycle_A_3

            if self.opt.fake_diff_loss or self.opt.real_fake_diff_loss:
                self.fake_B_diff = torch.sub(self.fake_B_1, self.fake_B_2, alpha=1)

            if self.opt.fake_diff_loss:
                # Dı' Ds difference matrix loss
                self.loss_fake_diff = self.criterionFakeDiff(self.fake_B_3, self.fake_B_diff) * lambda_fake_diff
                self.loss_G += self.loss_fake_diff

            if self.opt.real_fake_diff_loss:
                # Dı Ds difference loss
                self.loss_real_fake_diff = self.criterionRealFakeDiff(self.real_A_3, self.fake_B_diff) * lambda_real_fake_diff
                self.loss_G += self.loss_real_fake_diff

            if self.opt.fake_rec_diff_loss:
                # Dı' Ds' difference matrix loss
                self.loss_fake_rec_diff = self.criterionFakeRecDiff(self.fake_B_3, self.rec_A_3) * lambda_fake_rec_diff
                self.loss_G += self.loss_fake_rec_diff

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
