import os
import sys

GPU_ID, MODEL, DATASET, FLAGNAME, *OPTIONS = sys.argv[1:]

# training settings
ROOT                = '/home/ubuntu/skt_research/deepfakes_datasets' #'/home/jwchoi/datasets/deepfakes_datasets'
model               = MODEL # 'xception' # 'resnet18' # 'xception'# ' vgg16' # 'EfficientNet-B0' # 'MesoInception4' # mobilenet_v2 # mobilenet_v3_small
dataset             = DATASET # 'FaceForensics,UADFV,DeepfakeTIMIT' # ALL # FaceForensics # UADFV # DeepfakeTIMIT # CelebDF # CelebDF,UADFV
seed                = 0
print_freq          = 5
val_freq            = 3000
num_workers         = 10

# hyper-prameter settings
## dataset
quality             = 100
img_size            = 299 # 224 # 256 # 299
train_frame_per_vid = 40
val_frame_per_vid   = 40
test_frame_per_vid  = 100
test_data_sample    = 10000
## trainning
epochs              = 10
max_iters           = 50000
batch_size          = 32 # 32 # 8
lr                  = 0.0002 # 0.0002 # 0.001 <- mi4
momentum            = 0.9
weight_decay        = 1e-5 # 1e-4 # 1e-5 <-mbv2,3
criterion           = 'CE'   # BCE # MSE
optim               = 'Adam' # SGD # RMSProp # AdamW
scheduler           = None #'StepLR' # None # CosineAnnealing
##

print(OPTIONS)
add_options = ''
for option in OPTIONS:
    option = option.strip()
    add_options+=' '+option
print(add_options)

# command
os.system('python train.py '+\
          f' --data_root {ROOT}'+\
          f' --print_freq {print_freq}'+\
          f' --val_freq {val_freq}'+\
          f' --gpu_idx {GPU_ID}'+\
          f' --seed {seed}'+\
          f' --flag {FLAGNAME}'+\
          f' --norecord'+\
          f' --verbose'+\
          #
          f' --model {model}'+\
          f' --pretrained'+\
          #
          # Method invariant feature learning
          # f' --phase_concat'+\
          #
          # Adversarial attack robustness
          # f' --oudefend'+\
          # f' --lambda_rect {0.0}'+\
          # f' --lipschitz'+\
          # f' --beta_lips {5}'+\
          # f' --psi_lips {1000}'+\
          #
          # Physical condition shift robustness
          # f' --self_supervised'+\
          #f' --fine_tuning'+\
          # f' --pretrain_already'+\
          # f' --pretrain_epochs {4}'+\
          # f' --n_views {2}'+\
          # f' --temperature {0.3}'+\
          #
          f' --dataset {dataset}'+\
          f' --quality {quality}'+\
          f' --img_size {img_size}'+\
          f' --train_sample_num {train_frame_per_vid}'+\
          f' --val_sample_num {val_frame_per_vid}'+\
          f' --test_sample_num {test_frame_per_vid}'+\
          f' --random_sample_num {test_data_sample}'+\
          f' --aug'+\
          f' --num_workers {num_workers}'+\
          #
          f' --epochs {epochs}'+\
          f' --max_iters {max_iters}'+\
          f' --batch_size {batch_size}'+\
          f' --lr {lr}'+\
          f' --momentum {momentum}'+\
          f' --criterion {criterion}'+\
          f' --weight_balance'+\
          f' --optim {optim}'+\
          f' --scheduler {scheduler}'+\
          f' --earlystop'+\
          add_options+\
          f' 2>&1 | tee logs/log_{FLAGNAME}'
          )

# os.system('python /home/jwchoi/jwchoi/etc/wait.py')
