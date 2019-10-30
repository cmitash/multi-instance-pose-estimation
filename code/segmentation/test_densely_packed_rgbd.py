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
depth_scale = 8000.0
num_images = 30
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

    for testcase in range(0,num_images):
        scene_folder = os.path.join(test_path, '%05d' % int(testcase))

        test_rgb_image = os.path.join(scene_folder, 'frame-000000.color.png')
        test_depth_image = os.path.join(scene_folder, 'frame-000000.depth.png')

        A_img_depth = cv2.imread(test_depth_image, cv2.IMREAD_UNCHANGED)
        A_img_depth = A_img_depth.astype(np.float32)
        A_img_depth = A_img_depth/depth_scale
        A_img_depth[A_img_depth > 2] = 0

        max_val = np.amax(A_img_depth[np.nonzero(A_img_depth)])
        min_val = np.amin(A_img_depth[np.nonzero(A_img_depth)])

        A_img_depth = (A_img_depth - min_val)*255.0/(max_val - min_val)
        A_img_depth[A_img_depth < 0] = 0

        A_img_depth = A_img_depth.astype(np.uint8)
        A_img_depth = np.expand_dims(A_img_depth, axis=2)


        A_img_rgb = Image.open(test_rgb_image).convert('RGB')
        A_img_rgb = np.array(A_img_rgb, dtype=np.uint8)

        A = np.concatenate((A_img_rgb, A_img_depth), axis=2)
        
        A = transforms.ToTensor()(A)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)

        A = A.unsqueeze(0)

        data = {'A': A, 'A_paths': scene_folder, 'object_classes': object_classes}
        model.set_input(data)
        model.test()

        print (test_rgb_image)