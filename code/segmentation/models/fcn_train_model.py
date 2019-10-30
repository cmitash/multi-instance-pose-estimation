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

class FCNTrainModel(BaseModel):
    def name(self):
        return 'FCNTrainModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['sem', 'edge', 'D_L']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['pred_label_viz', 'gt_label', 'pred_edge_viz', 'gt_edge_label','rgb_image','depth_image']

        self.visual_names = visual_names_A

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_L']

        # CM: Check initialization Build network
        self.netD_L = networks.define_D_L(opt.input_nc, opt.output_nc, self.gpu_ids) #define net layers

        if self.isTrain:
            # CM: Check initialization                                    
            self.optimizer_D_L = torch.optim.SGD(self.netD_L.parameters(), lr=opt.lr, momentum=0.99, weight_decay=5e-4)
            self.optimizers = []
            self.optimizers.append(self.optimizer_D_L)

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
        self.image = input['A'].to(self.device)
        self.class_label = input['A_label'].to(self.device)
        self.edge_label = input['A_edge'].to(self.device)
        self.image_paths = input['A_paths']
        self.rgb_image = input['A'][:,:3].to(self.device)
        self.depth_image = input['A'][:,3].unsqueeze(1).to(self.device)

    def forward(self):
        self.data, self.target_label, self.target_edge = Variable(self.image.cuda()), Variable(self.class_label.cuda()), Variable(self.edge_label.cuda())
        self.pred_label, self.pred_edge = self.netD_L(self.data)

        self.gt_label = self.class_label[0].cpu().numpy()
        self.gt_label = self.label_to_rgb(self.gt_label)

        self.gt_edge_label = self.edge_label[0].cpu().numpy()
        self.gt_edge_label = self.label_to_rgb(self.gt_edge_label)


        self.pred_label_viz = self.pred_label.data.max(1)[1].cpu().numpy()
        self.pred_label_viz = self.pred_label_viz[0]
        self.pred_label_viz = self.label_to_rgb(self.pred_label_viz)

        self.pred_edge_viz = self.pred_edge.data.max(1)[1].cpu().numpy()
        self.pred_edge_viz = self.pred_edge_viz[0]
        self.pred_edge_viz = self.label_to_rgb(self.pred_edge_viz)

    def backward_D_L(self):
        self.loss_sem = networks.cross_entropy2d(input = self.pred_label, target = self.target_label, ignore_index1=10)

        edge_indices = np.count_nonzero(self.edge_label)
        non_edge_indices = np.count_nonzero(self.edge_label == 0)

        ratio = edge_indices/non_edge_indices

        weight_edge = torch.tensor([ratio, 1.0]).cuda()
        # weight_edge = torch.tensor([1.0, 1.0]).cuda()
        self.loss_edge = networks.cross_entropy2d(input = self.pred_edge, target = self.target_edge, weight = weight_edge)

        self.loss_D_L = self.loss_sem + self.loss_edge
        # self.loss_D_L = self.loss_sem
        self.loss_D_L.backward()

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
        self.freeze_encoder_features()

        # forward
        self.forward()
        self.optimizer_D_L.zero_grad()
        self.backward_D_L()
        self.optimizer_D_L.step()
