from .base_model import BaseModel
from . import networks

import shutil
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os
import cv2

import seaborn as sns

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

class FCNSaveModel(BaseModel):
    def name(self):
        return 'FCNSaveModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['image', 'predictions', 'predictions_edge']
        # self.visual_names = ['image', 'predictions', 'gt_label']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_L']

        self.n_class = opt.output_nc  # not used in FCN8s
        self.hist = np.zeros((self.n_class, self.n_class))
        self.hist_edge = np.zeros((2,2))
        # self.netD_L = networks.define_D_L(opt.input_nc, opt.output_nc, self.n_class, self.gpu_ids)
        self.netD_L = networks.define_D_L(opt.input_nc, opt.output_nc, self.gpu_ids)

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
        self.image_paths = input['A_paths']
        self.object_map = input['object_classes']


    def forward(self):
        self.data = Variable(self.image.cuda())
        self.pred_label, self.pred_edge = self.netD_L(self.data)

        self.predictions = self.pred_label.data.max(1)[1].cpu().numpy()
        self.predictions = self.predictions[0]

        self.predictions_edge = self.pred_edge.data.max(1)[1].cpu().numpy()
        self.predictions_edge = self.predictions_edge[0]

        print('self.predictions shape = ',self.predictions.data.shape)
        print('self.predictions_edge shape = ',self.predictions_edge.shape)
        
        self.pred_label = self.pred_label.squeeze()
        self.pred_label = self.pred_label.data.cpu().numpy()

        prob_map_path = os.path.join(self.image_paths, 'probability_maps')
        if os.path.exists(prob_map_path):
            shutil.rmtree(prob_map_path)
        os.mkdir(prob_map_path)

        for classVal in range(0, self.n_class):
            class_prob = self.pred_label[classVal,:,:]
            super_threshold_indices = class_prob < 0
            class_prob[super_threshold_indices] = 0
            class_max = np.ndarray.max(class_prob)
            class_prob = class_prob/class_max
            class_prob = class_prob*10000
            class_prob_int = class_prob.astype(np.uint16)
            cv2.imwrite(os.path.join(self.image_paths,'probability_maps/%s.png' % (self.object_map[classVal])), class_prob_int)
            self.saveHeatmap(class_prob_int, os.path.join(self.image_paths,'probability_maps/%s_heatmap.png' % (self.object_map[classVal])))

        self.softmaxHeatmap()
        print('self.heatmap shape: ', self.heatmap.shape)
        self.heatmap_edge_dim = self.heatmap[1]

        # original code
        # self.heatmap = self.heatmap[1] * 10000 #edges
        # self.heatmap = np.array(self.heatmap, dtype=np.uint16)
        # cv2.imwrite(os.path.join(self.image_paths,'probability_maps/edge.png'), self.heatmap)
        # self.saveHeatmap(self.heatmap, os.path.join(self.image_paths,'probability_maps/edge_heatmap.png'))
        
        #let's try thresholding
        self.heatmap = self.heatmap[1] * 255
        self.heatmap = np.array(self.heatmap, dtype=np.uint8)
        ret, self.heatmap = cv2.threshold(self.heatmap, 102, 255, cv2.THRESH_BINARY_INV) # 102 corresponds to 0.4 probability in 0-255
        cv2.imwrite(os.path.join(self.image_paths,'probability_maps/edge.png'), self.heatmap)

        #============== visualization =========
        self.predictions = self.label_to_rgb(self.predictions)
        self.predictions_edge = self.label_to_rgb(self.predictions_edge)


    def softmaxHeatmap(self):
        self.heatmap = self.pred_edge.data[0]
        self.heatmap = F.softmax(self.heatmap,dim=0)

    def sigmoidHeatmap(self):
        self.heatmap = self.pred_edge.data[0].sigmoid();

    def saveHeatmap(self, img, save_location):
        plt.clf()
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        ax = sns.heatmap(img, cmap=plt.get_cmap('plasma'))
        plt.savefig(save_location)