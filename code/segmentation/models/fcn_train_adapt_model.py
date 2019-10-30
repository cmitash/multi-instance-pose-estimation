import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from PIL import Image
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class FCNTrainAdaptModel(BaseModel):
    def name(self):
        return 'FCNTrainAdaptModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['sem', 'edge', 'D_L', 'D_L_domain_class', 'D_L_domain_edge', 'D_domain','D_domain_edge']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # visual_names_A = ['gt_label', 'gt_edge_label','syn_rgb_image','syn_depth_image', 'pred_syn_edge_viz', 'pred_syn_class_viz', 'real_rgb_image','real_depth_image','pred_real_edge_viz','pred_real_class_viz']
        visual_names_A = ['gt_label', 'gt_edge_label','syn_rgb_image', 'pred_syn_edge_viz', 'pred_syn_class_viz', 'real_rgb_image','pred_real_edge_viz','pred_real_class_viz']

        self.visual_names = visual_names_A

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_L', 'D_SR', 'D_SR_edge']

        # CM: Check initialization Build network
        self.netD_L = networks.define_D_L(opt.input_nc, opt.output_nc, self.gpu_ids) #define net layers
        use_sigmoid = False
        self.netD_SR = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        self.netD_SR_edge = networks.define_D(2, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        if self.isTrain:
            # CM: Check initialization
            self.optimizer_D_SR = torch.optim.Adam(self.netD_SR.parameters(), lr=2e-3, betas=(opt.beta1, 0.99))                                   
            self.optimizer_D_SR_edge = torch.optim.Adam(self.netD_SR_edge.parameters(), lr=2e-3, betas=(opt.beta1, 0.99))     
            self.optimizer_D_L = torch.optim.SGD(self.netD_L.parameters(), lr=opt.lr, momentum=0.99, weight_decay=5e-4)
            self.optimizers = []
            self.optimizers.append(self.optimizer_D_L)
            self.optimizers.append(self.optimizer_D_SR)
            self.optimizers.append(self.optimizer_D_SR_edge)

    def label_to_rgb(self, label):
        class2rgb = [[0, 0, 0], #Ignore 
        [128,0,0], #Background
        [0,128,0], #Wall
        [128,128,0], #Floor
        [0,0,128], #Ceiling
        [128,0,128], #Table
        [0,128,128], #Chair
        [128,128,128], #Window
        [64,0,0], #Door
        [192,0,0], #Monitor
        [64, 128, 0]  # not used
        ]

        h, w = label.shape  #img size
        color_label = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                class_idx = int(label[i,j])
                color_label[i,j,0] = class2rgb[class_idx][0]
                color_label[i,j,1] = class2rgb[class_idx][1]
                color_label[i,j,2] = class2rgb[class_idx][2]

        color_label = transforms.ToTensor()(color_label)
        color_label = color_label.unsqueeze(0)  # shape(1,h,w,3)

        return color_label
                
    def set_input(self, input):
        self.syn_image = input['A'].to(self.device)
        self.real_image = input['B'].to(self.device)

        self.class_label = input['A_label'].to(self.device)

        self.edge_label = input['A_edge'].to(self.device)

        self.syn_image_paths = input['A_paths']
        self.real_image_paths = input['B_paths']

        self.syn_rgb_image = input['A'][:,:3].to(self.device)
        # self.syn_depth_image = input['A'][:,3].unsqueeze(1).to(self.device)

        self.real_rgb_image = input['B'][:,:3].to(self.device)
        # self.real_depth_image = input['B'][:,3].unsqueeze(1).to(self.device)
        

    # TODO
    def freeze_encoder_features(self):
        child_counter = 0
        for child in self.netD_L.children():
            for fcn_layer in child.children():
                if isinstance(fcn_layer,nn.BatchNorm2d):
                    continue
                if child_counter < 31:
                    # print ("fcn layer ", child_counter, "is:")
                    # print(fcn_layer)
                    for params in fcn_layer.parameters():
                        params.requires_grad = False
                child_counter += 1
            break

    def optimize_parameters(self):  #update params
        # self.freeze_encoder_features()  ## change this when finetune

        #======================= segmentation and edge loss =====================
        self.syn_data, self.real_data, self.target_label, self.target_edge = Variable(self.syn_image.cuda()), Variable(self.real_image.cuda()), Variable(self.class_label.cuda()), Variable(self.edge_label.cuda())
        self.pred_syn_class, self.pred_syn_edge = self.netD_L(self.syn_data)

        self.gt_label = self.class_label[0].cpu().numpy()
        self.gt_label = self.label_to_rgb(self.gt_label)
        self.edge_label = self.edge_label[0].cpu().numpy()
        self.gt_edge_label = self.label_to_rgb(self.edge_label)

        self.optimizer_D_L.zero_grad()
        self.set_requires_grad([self.netD_SR], False) # Need to define D_SR
        self.set_requires_grad([self.netD_SR_edge], False)

        self.loss_sem = networks.cross_entropy2d(input = self.pred_syn_class, target = self.target_label, ignore_index1=10)

        edge_indices = np.count_nonzero(self.edge_label)
        non_edge_indices = np.count_nonzero(self.edge_label == 0)

        ratio = edge_indices/non_edge_indices

        weight_edge = torch.tensor([ratio, 1.0]).cuda()
        self.loss_edge = networks.cross_entropy2d(input = self.pred_syn_edge, target = self.target_edge, weight = weight_edge)

        self.loss_D_L = self.loss_sem + self.loss_edge
        self.loss_D_L.backward()

        #================== update G =======================
        self.pred_real_class, self.pred_real_edge = self.netD_L(self.real_data)
        self.D_out = self.netD_SR(F.softmax(self.pred_real_class, dim=1))

        # ## NOTE: D input dim=3, to make edge feedable, we concatenate a zero tensor in the 3rd dim
        # zero_tensor = torch.zeros((self.pred_real_edge.shape[0],1,self.pred_real_edge.shape[2],self.pred_real_edge.shape[3])).cuda()
        # pred_real_edge = torch.cat( (self.pred_real_edge, zero_tensor), 1)

        self.D_out_edge = self.netD_SR_edge(F.softmax(self.pred_real_edge, dim=1))
        self.loss_D_L_domain_class = self.bce_loss(self.D_out, Variable(torch.FloatTensor(self.D_out.data.size()).fill_(0)).cuda())
        self.loss_D_L_domain_edge = self.bce_loss(self.D_out_edge, Variable(torch.FloatTensor(self.D_out_edge.data.size()).fill_(0)).cuda())

        self.domain_loss_weight = 0.001
        self.loss_D_L_domain_weighted = self.domain_loss_weight * 0.5 * (self.loss_D_L_domain_class+self.loss_D_L_domain_edge)
        self.loss_D_L_domain_weighted.backward()
        self.optimizer_D_L.step()

        #==================== update D ============================
        self.set_requires_grad([self.netD_SR], True)
        self.set_requires_grad([self.netD_SR_edge], True)
        self.optimizer_D_SR.zero_grad()
        self.optimizer_D_SR_edge.zero_grad()
        pred_syn_class = self.pred_syn_class.detach()
        pred_real_class = self.pred_real_class.detach()
        pred_syn_class = F.softmax(pred_syn_class, dim = 1)
        pred_real_class = F.softmax(pred_real_class, dim = 1)
        self.D_syn = self.netD_SR(pred_syn_class)
        self.D_real = self.netD_SR(pred_real_class)

        pred_syn_edge = self.pred_syn_edge.detach()
        pred_real_edge = self.pred_real_edge.detach()
        pred_syn_edge = F.softmax(pred_syn_edge, dim = 1)
        pred_real_edge = F.softmax(pred_real_edge, dim = 1)
        # pred_real_edge = torch.cat( (pred_real_edge, zero_tensor), 1)
        # pred_syn_edge = torch.cat( (pred_syn_edge, zero_tensor), 1)
        self.D_syn_edge = self.netD_SR_edge(pred_syn_edge)
        self.D_real_edge = self.netD_SR_edge(pred_real_edge)

        syn_domain_gt = Variable(torch.FloatTensor(self.D_syn.size()).fill_(0)).cuda()
        real_domain_gt = Variable(torch.FloatTensor(self.D_real.size()).fill_(1)).cuda()

        syn_edge_domain_gt = Variable(torch.FloatTensor(self.D_syn_edge.size()).fill_(0)).cuda()
        real_edge_domain_gt = Variable(torch.FloatTensor(self.D_real_edge.size()).fill_(1)).cuda()

    
        self.D_syn_loss = self.bce_loss(self.D_syn, syn_domain_gt)
        self.D_real_loss = self.bce_loss(self.D_real, real_domain_gt)

        self.D_syn_edge_loss = self.bce_loss(self.D_syn_edge, syn_edge_domain_gt)
        self.D_real_edge_loss = self.bce_loss(self.D_real_edge, real_edge_domain_gt)

        self.loss_D_domain = 0.5*(self.D_syn_loss + self.D_real_loss)
        self.loss_D_domain.backward()

        self.loss_D_domain_edge = 0.5*(self.D_syn_edge_loss + self.D_real_edge_loss)
        self.loss_D_domain_edge.backward()
        
        self.optimizer_D_SR.step()
        self.optimizer_D_SR_edge.step()


        #=================== VIZ =====================================
        self.pred_syn_class = self.pred_syn_class.data.max(1)[1].cpu().numpy()
        self.pred_syn_class = self.pred_syn_class[0]
        self.pred_syn_class_viz = self.label_to_rgb(self.pred_syn_class)

        self.pred_syn_edge = self.pred_syn_edge.data.max(1)[1].cpu().numpy()
        self.pred_syn_edge = self.pred_syn_edge[0]
        self.pred_syn_edge_viz = self.label_to_rgb(self.pred_syn_edge)

        self.pred_real_class = self.pred_real_class.data.max(1)[1].cpu().numpy()
        self.pred_real_class = self.pred_real_class[0]
        self.pred_real_class_viz = self.label_to_rgb(self.pred_real_class)

        self.pred_real_edge = self.pred_real_edge.data.max(1)[1].cpu().numpy()
        self.pred_real_edge = self.pred_real_edge[0]
        self.pred_real_edge_viz = self.label_to_rgb(self.pred_real_edge)
