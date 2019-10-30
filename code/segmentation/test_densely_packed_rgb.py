import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import cv2

object_classes = {0: 'background', 1: 'dove', 2: 'toothpaste'}

test_path = 'datasets/densely_packed_dataset/'

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    model = create_model(opt)
    model.setup(opt)

    for testcase in range(0, 30):
        scene_folder = os.path.join(test_path, '%05d' % int(testcase))

        test_rgb_image = os.path.join(scene_folder, 'frame-000000.color.png')

        print (test_rgb_image)

        A_img_rgb = Image.open(test_rgb_image).convert('RGB')

        A = transforms.ToTensor()(A_img_rgb)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)

        A = A.unsqueeze(0)

        data = {'A': A, 'A_paths': scene_folder, 'object_classes': object_classes}
        model.set_input(data)
        model.test()