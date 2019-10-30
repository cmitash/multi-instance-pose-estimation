# Semantic and instance boundary segmentation with GAN adaptation

## Dependencies:
dominate  
pytorch  
torchvision  
visdom  

# Start Visdom for visualization 
python -m visdom.server
(Training output can be seen at localhost:8097)

# Train FCN GAN-ADAPT
python train.py --dataroot ./datasets/densely_packed_training/ --model fcn_train_adapt --name densely_packed --dataset_mode gan --display_freq 1 --lr 0.0002 --gpu_ids 0 --batchSize 4  --output_nc 3 --fineSize 480

# Test FCN GAN-ADAPT
python test_jdx_rgb.py --dataroot ./datasets/densely_packed_test/ --model fcn_save --name densely_packed --dataset_mode single --input_nc 3 --output_nc 3

# Test an already trained model
Download and place the model file in a directory within the checkpoints directory and specify the directory name by passing to the 'name' parameter at test time. 

# Code citation
The code is developed over the framework shared by:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix