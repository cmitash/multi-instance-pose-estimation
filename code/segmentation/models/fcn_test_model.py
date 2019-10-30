from .base_model import BaseModel
from . import networks

import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def _fast_hist(label_true, label_pred, n_class, usemask=True):
    if usemask:
        # ignoring class 0
        mask = (label_true > 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        # print('hist=\n',hist)
    else:
        hist = np.bincount(
            n_class * label_true.astype(int) +
            label_pred, minlength=n_class ** 2).reshape(n_class, n_class)        
    return hist

class FCNTestModel(BaseModel):
    def name(self):
        return 'FCNTestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['image', 'predictions', 'predictions_edge', 'heatmap']
        # self.visual_names = ['image', 'predictions', 'gt_label']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_L']

        self.n_class = 3  # not used in FCN8s
        self.hist = np.zeros((self.n_class, self.n_class))
        self.hist_edge = np.zeros((2,2))
        self.netD_L = networks.define_D_L(opt.input_nc, opt.output_nc, self.n_class, self.gpu_ids)

    def label_accuracy_score(self, label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.

        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        - precision
        - recall
        """
        for lt, lp in zip(label_trues, label_preds):
            self.hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    def edge_accuracy_score(self, label_trues, label_preds):
        """Returns accuracy score evaluation result.

        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        - precision
        - recall
        """
        for lt, lp in zip(label_trues, label_preds):
            self.hist_edge += _fast_hist(lt.flatten(), lp.flatten(), 2, False)

    def get_stats(self):
        
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        prec = np.diag(self.hist)/(self.hist.sum(axis=0))
        rec = np.diag(self.hist)/(self.hist.sum(axis=1))
        # print (iu)
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc, prec, rec

    def get_stats_edge(self):
        prec = np.diag(self.hist_edge)/(self.hist_edge.sum(axis=0))
        rec = np.diag(self.hist_edge)/(self.hist_edge.sum(axis=1))
        return prec, rec

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
        [192,0,0] #Monitor
        ]

        h, w = label.shape
        color_label = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                class_idx = int(label[i,j])
                color_label[i,j,0] = class2rgb[class_idx][0]
                color_label[i,j,1] = class2rgb[class_idx][1]
                color_label[i,j,2] = class2rgb[class_idx][2]

        color_label = transforms.ToTensor()(color_label)
        color_label = color_label.unsqueeze(0)

        return color_label

    def set_input(self, input):
        self.image = input['A'].to(self.device)
        self.class_label = input['A_label'].to(self.device)
        self.edge_label = input['A_edge'].to(self.device)
        self.image_paths = input['A_paths']


    def forward(self):
        self.data = Variable(self.image.cuda())
        self.pred_label, self.pred_edge = self.netD_L(self.data)

        self.gt_label = self.class_label[0].cpu().numpy()
        self.gt_edge_label = self.edge_label[0].cpu().numpy()

        

        self.predictions = self.pred_label.data.max(1)[1].cpu().numpy()
        self.predictions = self.predictions[0]
        self.predictions_edge = self.pred_edge.data.max(1)[1].cpu().numpy()
        self.predictions_edge = self.predictions_edge[0]

        print('self.predictions shape = ',self.predictions.data.shape)
        print('self.predictions_edge shape = ',self.predictions_edge.shape)
        print('self.gt_label shape = ',self.gt_label.shape)
        print('self.gt_edge_label shape = ',self.gt_edge_label.shape)
        

        self.softmaxHeatmap()
        print('self.heatmap shape: ', self.heatmap.shape)
        self.heatmap_edge_dim = self.heatmap[1]
        self.heatmap = self.heatmap[1] * 255 #edges
        self.heatmap = np.array(self.heatmap, dtype=np.uint8) 
        tmp=np.zeros((self.heatmap.shape[0],self.heatmap.shape[1],3))
        tmp[:,:,0]=self.heatmap
        self.heatmap=tmp
        self.heatmap = transforms.ToTensor()(self.heatmap)
        self.heatmap = self.heatmap.unsqueeze(0)

        
        use_thres=False
        if use_thres:
            self.predictions_edge = np.array(self.heatmap_edge_dim,dtype=np.float32)
            mask = self.predictions_edge > 0.8
            
            
            self.predictions_edge = np.zeros(self.predictions_edge.shape)

            self.predictions_edge[mask] = 1
            self.predictions_edge = np.array(self.predictions_edge, dtype=np.int8)
            

        #=============== eval ==================
        self.label_accuracy_score(self.gt_label, self.predictions, self.n_class)
        self.edge_accuracy_score(self.gt_edge_label, self.predictions_edge)


        #============== visualization =========
        self.predictions = self.label_to_rgb(self.predictions)
        self.predictions_edge = self.label_to_rgb(self.predictions_edge)


    def softmaxHeatmap(self):
        self.heatmap = self.pred_edge.data[0]
        self.heatmap = F.softmax(self.heatmap,dim=0)

    def sigmoidHeatmap(self):
        self.heatmap = self.pred_edge.data[0].sigmoid();
