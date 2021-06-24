import os
import sys

GPU_ID              = sys.argv[1]
CKPT_PATH           = sys.argv[2]

# test settings
ROOT                = '/home/jwchoi/datasets/deepfakes_datasets'
model               = 'xception' # 'mobilenet_v3_small' # 'EfficientNet-B0' # 'mobilenet_v2' # 'resnet18' # 'xception' # 'vgg16' # 'MesoInception4'
dataset             = 'FaceForensics' # ALL # FaceForensics # UADFV # DeepfakeTIMIT # CelebDF # CelebDF,UADFV
seed                = 0
num_workers         = 20
batch_size          = 128
quality             = 100
img_size            = 299 # 224 # 256 # 299
# val_sample_num    = 40
test_sample_num     = 100
random_sample_num   = 10000

# command
os.system('python test.py '+\
          ' --data_root {}'.format(ROOT)+\
          ' --ckpt_path {}'.format(CKPT_PATH)+\
          ' --gpu_idx {}'.format(GPU_ID)+\
          ' --seed {}'.format(seed)+\
          #
          ' --model {}'.format(model)+\
          ' --batch_size {}'.format(batch_size)+\
          #
          ' --dataset {}'.format(dataset)+\
          ' --quality {}'.format(quality)+\
          ' --img_size {}'.format(img_size)+\
          # ' --val_sample_num {}'.format(test_sample_num)+\
          ' --test_sample_num {}'.format(test_sample_num)+\
          ' --random_sample_num {}'.format(random_sample_num)+\
          ' --num_workers {}'.format(num_workers)+\
          ' --test_only'+\
          ' --verbose'
          )
