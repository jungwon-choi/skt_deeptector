import os
import sys

GPU_ID              = sys.argv[1]   # GPU Index
FLAGNAME            = sys.argv[2]   # flag for saved files

# training settings
ROOT                = '/home/jwchoi/datasets/deepfakes_datasets'
model               = 'xception' # 'resnet18' # 'xception'# ' vgg16' # 'EfficientNet-B0' # 'MesoInception4' # mobilenet_v2 # mobilenet_v3_small
dataset             = 'FaceForensics' # ALL # FaceForensics # UADFV # DeepfakeTIMIT # CelebDF # CelebDF,UADFV
seed                = 0
print_freq          = 5
val_freq            = 500
num_workers         = 8

# hyper-prameter settings
## dataset
quality             = 100
img_size            = 299 # 224 # 256 # 299
train_sample_num    = 40
val_sample_num      = 40
test_sample_num     = 100
random_sample_num   = 5000
## trainning
epochs              = 10
max_iters           = 3000
batch_size          = 32
lr                  = 0.0002 # 0.001
momentum            = 0.9
weight_decay        = 1e-4
criterion           = 'CE'   # BCE # MSE
optim               = 'Adam' # SGD # RMSProp # AdamW
scheduler           = 'StepLR' # None # CosineAnnealing

# command
os.system('python train.py '+\
          ' --data_root {}'.format(ROOT)+\
          ' --print_freq {}'.format(print_freq)+\
          ' --val_freq {}'.format(val_freq)+\
          ' --gpu_idx {}'.format(GPU_ID)+\
          ' --seed {}'.format(seed)+\
          ' --flag {}'.format(FLAGNAME)+\
          ' --norecord'+\
          ' --verbose'+\
          #
          ' --model {}'.format(model)+\
          ' --pretrained'+\
          #
          ' --dataset {}'.format(dataset)+\
          ' --quality {}'.format(quality)+\
          ' --img_size {}'.format(img_size)+\
          ' --train_sample_num {}'.format(train_sample_num)+\
          ' --val_sample_num {}'.format(val_sample_num)+\
          ' --test_sample_num {}'.format(test_sample_num)+\
          ' --random_sample_num {}'.format(random_sample_num)+\
          ' --aug'+\
          ' --num_workers {}'.format(num_workers)+\
          #
          ' --epochs {}'.format(epochs)+\
          ' --max_iters {}'.format(max_iters)+\
          ' --batch_size {}'.format(batch_size)+\
          ' --lr {}'.format(lr)+\
          ' --momentum {}'.format(momentum)+\
          ' --criterion {}'.format(criterion)+\
          ' --weight_balance'+\
          ' --optim {}'.format(optim)+\
          ' --scheduler {}'.format(scheduler)+\
          ' --earlystop'
          )
