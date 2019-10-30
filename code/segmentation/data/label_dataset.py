import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import random
import numpy as np

class LabelDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot,  'rgb')
        self.dir_edge_A = os.path.join(opt.dataroot,'edge_labels')
        self.dir_label_A = os.path.join(opt.dataroot, 'seg_labels')
        

        self.A_paths = make_dataset(self.dir_A)
        self.A_label_paths = make_dataset(self.dir_label_A)
        self.A_edge_paths = make_dataset(self.dir_edge_A)

        self.A_paths = sorted(self.A_paths)
        self.A_label_paths = sorted(self.A_label_paths)
        self.A_edge_paths = sorted(self.A_edge_paths)
        self.A_size = len(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A_label_path = self.A_label_paths[index % self.A_size]
        A_edge_path = self.A_edge_paths[index % self.A_size]

        A_img = Image.open(A_path).convert('RGB')
        A_label_img = Image.open(A_label_path).convert('P')
        A_edge_img = Image.open(A_edge_path)
        
        A = A_img
        A_label = A_label_img
        A_edge = A_edge_img

        A = transforms.ToTensor()(A)
        A_label = np.array(A_label, dtype=np.uint8)
        A_edge = np.array(A_edge, dtype=np.uint8)

        w_offset = random.randint(0, max(0, 640 - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, 480 - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        A_label = A_label[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        A_edge = A_edge[h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]


        A_edge_indices = A_edge > 0
        A_edge[A_edge_indices] = 1

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)

        input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
	
        A_label = torch.from_numpy(A_label).long()
        A_edge = torch.from_numpy(A_edge).long()

        return {'A': A, 'A_label': A_label, 'A_edge': A_edge,
                'A_paths': A_path, 'A_label_paths': A_label_path, 'A_edge_paths': A_edge_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'LabelDataset'
