import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from PIL import Image
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms

class CycleGANSemanticModel(BaseModel):
    def name(self):
        return 'CycleGANSemanticModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'sem_AB', 'D_L']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # visual_names_A = ['real_A', 'fake_B', 'rec_A']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']
        
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = []

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_L']
        self.preload_model_names = ['D_L']

        self.loss_G_A = 0
        self.loss_G_B = 0
        self.loss_D_A = 0
        self.loss_D_B = 0
        self.loss_sem_AB = 0
        self.loss_sem_BA = 0
        self.loss_cycle_A = 0
        self.loss_cycle_B = 0
        self.loss_D_L = 0
        
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        use_sigmoid = opt.no_lsgan
        self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        # CM: Number of semantic classes
        self.netD_L = networks.define_D_L(opt.input_nc, opt.output_nc, self.gpu_ids)

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

        # CM: Learning rate for FCN is hard-coded over here as it could be different from the learning rate of GAN
        self.optimizer_D_L = torch.optim.SGD(self.netD_L.parameters(),
                                            lr=1e-3, momentum=0.9, weight_decay=5e-4)
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.input_A_label = input['A_label'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        # get label prediction
        pred_real_B, _ = self.netD_L(Variable(self.real_B.cuda()))
        _,self.pred_real_B = pred_real_B.max(1)

        self.pred_fake_A, _ = self.netD_L(Variable(self.fake_A.cuda()))
        self.pred_fake_B, _ = self.netD_L(Variable(self.fake_B.cuda()))

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

    def backward_D_L(self):
        pred_fake_B, _ = self.netD_L(Variable(self.fake_B.cuda())) 
        self.loss_D_L = networks.cross_entropy2d(input = pred_fake_B, target = self.input_A_label, ignore_index1=10)
        self.loss_D_L.backward()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B

        # semantic loss AB
        self.loss_sem_AB = networks.cross_entropy2d(input = self.pred_fake_B, target = self.input_A_label, ignore_index1=10)

        self.loss_G += self.loss_sem_AB
        self.loss_G.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_D_L.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

        # else:
        #     self.set_requires_grad([self.netG_A, self.netG_B], False)
        #     self.set_requires_grad([self.netD_A, self.netD_B], False)
        #     self.set_requires_grad([self.netD_L], True)

        #     self.fake_B = self.netG_A(self.real_A)
        #     self.pred_fake_B = self.netD_L(Variable(self.fake_B.cuda()))

        #     self.optimizer_D_L.zero_grad()
        #     self.backward_D_L()
        #     self.optimizer_D_L.step()