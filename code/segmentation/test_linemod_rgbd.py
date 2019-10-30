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

object_classes = {0: 'background', 1: 'obj_01', 2: 'obj_02', 3: 'obj_05',
            4: 'obj_06', 5: 'obj_08', 6: 'obj_09', 7: 'obj_10',
            8: 'obj_11', 9: 'obj_12'}

test_scenes = [3, 8, 17, 27, 36, 38, 39, 41, 47, 58, 61, 62, 64, 65, 69, 72, 79, 89, 96, 97, 102, 107, 110, 115, 119, 124, 126, 136, 153, 156, 162, 166, 175, 176, 178, 203, 207, 217, 219, 221, 224, 243, 248, 249, 254, 258, 263, 266, 268, 277, 283, 307, 310, 322, 326, 338, 342, 356, 362, 365, 368, 387, 389, 402, 415, 417, 425, 428, 434, 435, 438, 442, 446, 453, 473, 474, 476, 477, 480, 491, 494, 499, 501, 503, 521, 527, 529, 532, 535, 540, 543, 549, 560, 563, 571, 575, 589, 603, 607, 611, 615, 625, 642, 648, 649, 650, 652, 667, 669, 679, 691, 695, 703, 708, 711, 727, 736, 737, 739, 740, 750, 754, 756, 757, 758, 761, 762, 764, 768, 769, 770, 773, 775, 785, 788, 791, 794, 801, 803, 804, 808, 809, 819, 821, 828, 837, 840, 844, 850, 856, 867, 871, 877, 883, 886, 894, 902, 903, 904, 907, 909, 918, 925, 934, 942, 952, 956, 961, 968, 969, 972, 982, 984, 991, 1001, 1012, 1038, 1050, 1057, 1061, 1069, 1071, 1087, 1098, 1099, 1103, 1107, 1117, 1123, 1131, 1144, 1148, 1151, 1157, 1168, 1169, 1176, 1180, 1199, 1212]
test_path = 'datasets/occ_linemod/'

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    model = create_model(opt)
    model.setup(opt)

    for testcase in test_scenes:
        scene_folder = os.path.join(test_path, '%04d' % int(testcase))

        test_rgb_image = os.path.join(scene_folder, 'rgb.png')
        test_depth_image = os.path.join(scene_folder, 'depth.png')

        A_img_depth = cv2.imread(test_depth_image, cv2.IMREAD_UNCHANGED)
        A_img_depth = A_img_depth.astype(np.float32)
        A_img_depth = A_img_depth/1000.0
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
        A = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(A)

        A = A.unsqueeze(0)

        data = {'A': A, 'A_paths': scene_folder, 'object_classes': object_classes}
        model.set_input(data)
        model.test()

        print (test_rgb_image)
