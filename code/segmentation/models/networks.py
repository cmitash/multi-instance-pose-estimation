import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from distutils.version import LooseVersion

import os.path as osp
import torchvision.models as models
import numpy as np

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
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


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], num_classes=10):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'prediction':
        netD = PredictionDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)

def define_D_L(input_nc, output_nc, gpu_ids=[], init_type='normal'):
    net = FCN8s(input_nc, output_nc)
    vgg16 = models.vgg16(pretrained=True)
    net.init_vgg16_params(vgg16)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    # init_weights(net, init_type)
    return net

##############################################################################
# Classes
##############################################################################

def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index1=None):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    if ignore_index1 is not None:
        loss = F.nll_loss(log_p, target, weight=weight, size_average=False, ignore_index=ignore_index1)
    else:
        loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()

    # print ('loss: ', loss)
    return loss
    
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

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FCN8s, self).__init__()
        
        # conv1  5
        self.conv1_1 = nn.Conv2d(input_nc, 64, 3, padding=1)
        self.conv1_1_BN = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_BN = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)  # 1/2

        # conv2   5
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_BN = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_BN = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)  # 1/4

        # conv3  7
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_BN = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_BN = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_BN = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)  # 1/8

        # conv4   7
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_BN = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_BN = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_BN = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)  # 1/16

        # conv5  7
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_BN = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_BN = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_BN = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)  # 1/32

        # # fc6
        # self.fc6 = nn.Conv2d(512, 4096, 7)
        # self.fc6_BN = nn.BatchNorm2d(4096)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.drop6 = nn.Dropout2d()

        # # fc7
        # self.fc7 = nn.Conv2d(4096, 4096, 1)
        # self.fc7_BN = nn.BatchNorm2d(4096)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()

        # # for semantic classes
        # self.deconv6_sem = nn.ConvTranspose2d(4096, 512, 7)
        # nn.init.xavier_uniform(self.deconv6_sem.weight)
        # self.deconv6_sem_BN = nn.BatchNorm2d(512)
        # self.relu6_d = nn.ReLU(inplace=True)
        self.up5_sem = nn.MaxUnpool2d(2, stride=2)
        self.deconv5_3 = nn.ConvTranspose2d(512, 512, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv5_3.weight)
        self.deconv5_3_BN = nn.BatchNorm2d(512)
        self.relu5_3d = nn.ReLU(inplace=True)
        self.deconv5_2 = nn.ConvTranspose2d(512, 512, 3, padding = 1)
        self.deconv5_2_BN = nn.BatchNorm2d(512)
        nn.init.xavier_uniform(self.deconv5_2.weight)
        self.relu5_2d = nn.ReLU(inplace=True)
        self.deconv5_1 = nn.ConvTranspose2d(512, 512, 3, padding = 1)
        self.deconv5_1_BN = nn.BatchNorm2d(512)
        nn.init.xavier_uniform(self.deconv5_1.weight)
        self.relu5_1d = nn.ReLU(inplace=True)

        self.up4_sem = nn.MaxUnpool2d(2, stride=2)
        self.deconv4_3 = nn.ConvTranspose2d(512, 512, 3, padding = 1)
        self.deconv4_3_BN = nn.BatchNorm2d(512)
        nn.init.xavier_uniform(self.deconv4_3.weight)
        self.relu4_3d = nn.ReLU(inplace=True)
        self.deconv4_2 = nn.ConvTranspose2d(512, 512, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv4_2.weight)
        self.deconv4_2_BN = nn.BatchNorm2d(512)
        self.relu4_2d = nn.ReLU(inplace=True)
        self.deconv4_1 = nn.ConvTranspose2d(512, 256, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv4_1.weight)
        self.deconv4_1_BN = nn.BatchNorm2d(256)
        self.relu4_1d = nn.ReLU(inplace=True)
        

        self.up3_sem = nn.MaxUnpool2d(2, stride=2)
        self.deconv3_3 = nn.ConvTranspose2d(256, 256, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv3_3.weight)
        self.deconv3_3_BN = nn.BatchNorm2d(256)
        self.relu3_3d = nn.ReLU(inplace=True)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv3_2.weight)
        self.deconv3_2_BN = nn.BatchNorm2d(256)
        self.relu3_2d = nn.ReLU(inplace=True)
        self.deconv3_1 = nn.ConvTranspose2d(256, 128, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv3_1.weight)
        self.deconv3_1_BN = nn.BatchNorm2d(128)
        self.relu3_1d = nn.ReLU(inplace=True)
        

        self.up2_sem = nn.MaxUnpool2d(2, stride=2)
        self.deconv2_2 = nn.ConvTranspose2d(128, 128, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv2_2.weight)
        self.deconv2_2_BN = nn.BatchNorm2d(128)
        self.relu2_2d = nn.ReLU(inplace=True)
        self.deconv2_1 = nn.ConvTranspose2d(128, 64, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv2_1.weight)
        self.deconv2_1_BN = nn.BatchNorm2d(64)
        self.relu2_1d = nn.ReLU(inplace=True)

        self.up1_sem = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_2 = nn.ConvTranspose2d(64, 64, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv1_2.weight)
        self.deconv1_2_BN = nn.BatchNorm2d(64)
        self.relu1_2d = nn.ReLU(inplace=True)
        self.deconv1_1 = nn.ConvTranspose2d(64, 32, 3, padding = 1)
        nn.init.xavier_uniform(self.deconv1_1.weight)
        self.deconv1_1_BN = nn.BatchNorm2d(32)
        self.relu1_1d = nn.ReLU(inplace=True)
        
        #segmentation output
        self.up_sem = nn.ConvTranspose2d(32, output_nc, 1)
        nn.init.xavier_uniform(self.up_sem.weight)

        # for edge detection
        # self.conv6_1 = nn.Conv2d(512, 4096, 7, padding=3)
        # self.conv6_1_BN = nn.BatchNorm2d(4096)
        # self.relu6_1 = nn.ReLU(inplace=True)
        # self.drop6_1 = nn.Dropout2d()

        # self.conv6_2 = nn.Conv2d(4096, 512, 1)
        # self.conv6_2_BN = nn.BatchNorm2d(512)
        # self.relu6_2 = nn.ReLU(inplace=True)
        # self.drop6_2 = nn.Dropout2d()

        #unpooling and deconvolution: do we need to specify output size to match the same as while maxpooling?
        self.up5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5 = nn.ConvTranspose2d(512, 512, 5, padding=2)
        nn.init.xavier_uniform(self.deconv5.weight)
        self.deconv5_BN = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.drop8 = nn.Dropout2d()

        self.up4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(512, 256, 5, padding=2)
        nn.init.xavier_uniform(self.deconv4.weight)
        self.deconv4_BN = nn.BatchNorm2d(256)
        self.relu9 = nn.ReLU(inplace=True)
        self.drop9 = nn.Dropout2d()

        self.up3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 5, padding=2)
        nn.init.xavier_uniform(self.deconv3.weight) 
        self.deconv3_BN = nn.BatchNorm2d(128)       
        self.relu10 = nn.ReLU(inplace=True)
        self.drop10 = nn.Dropout2d()

        self.up2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, padding=2)
        nn.init.xavier_uniform(self.deconv2.weight)  
        self.deconv2_BN = nn.BatchNorm2d(64)      
        self.relu11 = nn.ReLU(inplace=True)
        self.drop11 = nn.Dropout2d()

        self.up1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, padding=2)
        nn.init.xavier_uniform(self.deconv1.weight)     
        self.deconv1_BN = nn.BatchNorm2d(32)       
        self.relu12 = nn.ReLU(inplace=True)
        self.drop12 = nn.Dropout2d()

        #edge output
        self.up_final = nn.ConvTranspose2d(32, 2, 5, padding=2)
        nn.init.xavier_uniform(self.up_final.weight)        
        
        self._initialize_weights()

    

    def _initialize_weights(self):
        fcn_layers = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
            # self.fc6, self.relu6,
            # self.fc7, self.relu7,
            # self.conv6_1, self.conv6_2,
            ]

        for m in fcn_layers:
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, ):
        h = x
        h = self.relu1_1(self.conv1_1_BN(self.conv1_1(h)))
        h = self.relu1_2(self.conv1_2_BN(self.conv1_2(h)))
        h, indices1 = self.pool1(h)
        pool1 = h

        h = self.relu2_1(self.conv2_1_BN(self.conv2_1(h)))
        h = self.relu2_2(self.conv2_2_BN(self.conv2_2(h)))
        h, indices2 = self.pool2(h)
        pool2 = h

        h = self.relu3_1(self.conv3_1_BN(self.conv3_1(h)))
        h = self.relu3_2(self.conv3_2_BN(self.conv3_2(h)))
        h = self.relu3_3(self.conv3_3_BN(self.conv3_3(h)))
        h, indices3 = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1_BN(self.conv4_1(h)))
        h = self.relu4_2(self.conv4_2_BN(self.conv4_2(h)))
        h = self.relu4_3(self.conv4_3_BN(self.conv4_3(h)))
        h, indices4 = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1_BN(self.conv5_1(h)))
        h = self.relu5_2(self.conv5_2_BN(self.conv5_2(h)))
        h = self.relu5_3(self.conv5_3_BN(self.conv5_3(h)))
        h, indices5 = self.pool5(h)
        pool5 = h

        # h = self.relu6(self.fc6_BN(self.fc6(h)))
        # h = self.drop6(h)

        # h = self.relu7(self.fc7_BN(self.fc7(h)))
        # h = self.drop7(h)

        # upsample for semantic classes
        # h = self.relu6_d(self.deconv6_sem_BN(self.deconv6_sem(h)))
        h = self.up5_sem(h, indices5, output_size=pool4.size())
        h =  self.relu5_3d(self.deconv5_3_BN(self.deconv5_3(h)))
        h =  self.relu5_2d(self.deconv5_2_BN(self.deconv5_2(h)))
        h =  self.relu5_1d(self.deconv5_1_BN(self.deconv5_1(h)))

        h = self.up4_sem(h, indices4, output_size=pool3.size())
        h =  self.relu4_3d(self.deconv4_3_BN(self.deconv4_3(h)))
        h =  self.relu4_2d(self.deconv4_2_BN(self.deconv4_2(h)))
        h =  self.relu4_1d(self.deconv4_1_BN(self.deconv4_1(h)))

        h = self.up3_sem(h, indices3, output_size=pool2.size())
        h =  self.relu3_3d(self.deconv3_3_BN(self.deconv3_3(h)))
        h =  self.relu3_2d(self.deconv3_2_BN(self.deconv3_2(h)))
        h =  self.relu3_1d(self.deconv3_1_BN(self.deconv3_1(h)))

        h = self.up2_sem(h, indices2, output_size=pool1.size())
        h =  self.relu2_2d(self.deconv2_2_BN(self.deconv2_2(h)))
        h =  self.relu2_1d(self.deconv2_1_BN(self.deconv2_1(h)))

        h = self.up1_sem(h, indices1)
        h = self.relu1_2d(self.deconv1_2_BN(self.deconv1_2(h)))
        h = self.relu1_1d(self.deconv1_1_BN(self.deconv1_1(h)))
        h_s = self.up_sem(h)

        # upsample for edge class

        h_e = pool5
        
        # h_e = self.relu6_1(self.conv6_1_BN(self.conv6_1(pool5)))
        # h_e = self.drop6_1(h_e)

        # h_e = self.relu6_2(self.conv6_2_BN(self.conv6_2(h_e)))
        # h_e = self.drop6_2(h_e)


        h_e = self.up5(h_e, indices5, output_size=pool4.size())
        h_e = self.relu8(self.deconv5_BN(self.deconv5(h_e)))
        self.drop8(h_e)

        h_e = self.up4(h_e, indices4, output_size=pool3.size())
        h_e = self.relu9(self.deconv4_BN(self.deconv4(h_e)))
        self.drop9(h_e)

        h_e = self.up3(h_e, indices3, output_size=pool2.size())
        h_e = self.relu10(self.deconv3_BN(self.deconv3(h_e)))
        self.drop10(h_e)

        h_e = self.up2(h_e, indices2, output_size=pool1.size())
        h_e = self.relu11(self.deconv2_BN(self.deconv2(h_e)))
        self.drop11(h_e)

        h_e = self.up1(h_e, indices1)
        h_e = self.relu12(self.deconv1_BN(self.deconv1(h_e)))
        self.drop12(h_e)

        h_e = self.up_final(h_e)
        return h_s, h_e  # prediction of segmentation and edge
        
        #     h = x
        #     h = self.relu1_1(self.conv1_1(h))
        #     h = self.relu1_2(self.conv1_2(h))
        #     h, indices1 = self.pool1(h)
        #     pool1 = h

        #     h = self.relu2_1(self.conv2_1(h))
        #     h = self.relu2_2(self.conv2_2(h))
        #     h, indices2 = self.pool2(h)
        #     pool2 = h

        #     h = self.relu3_1(self.conv3_1(h))
        #     h = self.relu3_2(self.conv3_2(h))
        #     h = self.relu3_3(self.conv3_3(h))
        #     h, indices3 = self.pool3(h)
        #     pool3 = h  # 1/8

        #     h = self.relu4_1(self.conv4_1(h))
        #     h = self.relu4_2(self.conv4_2(h))
        #     h = self.relu4_3(self.conv4_3(h))
        #     h, indices4 = self.pool4(h)
        #     pool4 = h  # 1/16

        #     h = self.relu5_1(self.conv5_1(h))
        #     h = self.relu5_2(self.conv5_2(h))
        #     h = self.relu5_3(self.conv5_3(h))
        #     h, indices5 = self.pool5(h)
        #     pool5 = h

        #     # h = self.relu6(self.fc6(h))
        #     # h = self.drop6(h)

        #     # h = self.relu7(self.fc7(h))
        #     # h = self.drop7(h)

        #     # upsample for semantic classes
        #     h = self.relu6_d(self.deconv6_sem(h))
        #     h = self.up5_sem(h, indices5, output_size=pool4.size())
        #     h =  self.relu5_3d(self.deconv5_3(h))
        #     h =  self.relu5_2d(self.deconv5_2(h))
        #     h =  self.relu5_1d(self.deconv5_1(h))

        #     h = self.up4_sem(h, indices4, output_size=pool3.size())
        #     h =  self.relu4_3d(self.deconv4_3(h))
        #     h =  self.relu4_2d(self.deconv4_2(h))
        #     h =  self.relu4_1d(self.deconv4_1(h))

        #     h = self.up3_sem(h, indices3, output_size=pool2.size())
        #     h =  self.relu3_3d(self.deconv3_3(h))
        #     h =  self.relu3_2d(self.deconv3_2(h))
        #     h =  self.relu3_1d(self.deconv3_1(h))

        #     h = self.up2_sem(h, indices2, output_size=pool1.size())
        #     h =  self.relu2_2d(self.deconv2_2(h))
        #     h =  self.relu2_1d(self.deconv2_1(h))

        #     h = self.up1_sem(h, indices1)
        #     h = self.relu1_2d(self.deconv1_2(h))
        #     h = self.relu1_1d(self.deconv1_1(h))
        #     h_s = self.up_sem(h)

        #     # upsample for edge class
        #     h_e = self.relu6_1(self.conv6_1(pool5))
        #     h_e = self.drop6_1(h_e)

        #     h_e = self.relu6_2(self.conv6_2(h_e))
        #     h_e = self.drop6_2(h_e)

        #     h_e = self.up5(h_e, indices5, output_size=pool4.size())
        #     h_e = self.relu8(self.deconv5(h_e))
        #     self.drop8(h_e)

        #     h_e = self.up4(h_e, indices4, output_size=pool3.size())
        #     h_e = self.relu9(self.deconv4(h_e))
        #     self.drop9(h_e)

        #     h_e = self.up3(h_e, indices3, output_size=pool2.size())
        #     h_e = self.relu10(self.deconv3(h_e))
        #     self.drop10(h_e)

        #     h_e = self.up2(h_e, indices2, output_size=pool1.size())
        #     h_e = self.relu11(self.deconv2(h_e))
        #     self.drop11(h_e)

        #     h_e = self.up1(h_e, indices1)
        #     h_e = self.relu12(self.deconv1(h_e))
        #     self.drop12(h_e)

        #     h_e = self.up_final(h_e)
        #     return h_s, h_e  # prediction of segmentation and edge

    def init_vgg16_params(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        isFirst = True
        for l1, l2 in zip(vgg16.features, features):
            if(isFirst):
               init.normal_(self.conv1_1.weight.data, 0.0, 0.02)
               isFirst = False
               continue
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                print(l1.weight.size(), l2.weight.size())
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        # for i, name in zip([0, 3], ['fc6', 'fc7']):
        #     l1 = vgg16.classifier[i]
        #     l2 = getattr(self, name)
        #     l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
        #     l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

# Discriminator for output space predictions
class PredictionDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        # number of classes is hardcoded
        super(PredictionDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


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
        # print(self.model)

    def forward(self, input):
        return self.model(input)


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

        print ('sigmoid: ', use_sigmoid)
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        # print(self.model)
        
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
