import os
import sys

GPU_ID, MODEL, DATASET, CKPT_PATH, *OPTIONS= sys.argv[1:]

# test settings
ROOT                = '/mnt/gsai/Datasets/deepfakes_datasets'#'/home/jwchoi/datasets/deepfakes_datasets'
model               = MODEL # 'mobilenet_v3_small' # 'mobilenet_v3_small' # 'EfficientNet-B0' # 'mobilenet_v2' # 'resnet18' # 'xception' # 'vgg16' # 'MesoInception4'
dataset             = DATASET #'FaceForensics,UADFV,DeepfakeTIMIT' # ALL # FaceForensics # UADFV # DeepfakeTIMIT # CelebDF # CelebDF,UADFV
seed                = 0
num_workers         = 10
batch_size          = 32
quality             = 100
# img_size            = 224 #299 # 224 # 256 # 299
# val_sample_num    = 40
test_sample_num     = 100
random_sample_num   = 10000

add_options = ''
for option in OPTIONS:
    add_options+=' '+option

# command
os.system('python attack_test.py '+\
          f' --data_root {ROOT}'+\
          f' --ckpt_path {CKPT_PATH}'+\
          f' --gpu_idx {GPU_ID}'+\
          f' --seed {seed}'+\
          #
          f' --model {model}'+\
          f' --batch_size {batch_size}'+\
          # (' --phase_concat' if CONCAT == 'True' else '')+\
          #
          f' --dataset {dataset}'+\
          f' --quality {quality}'+\
          # f' --img_size {img_size}'+\
          # f' --val_sample_num {test_sample_num}'+\
          f' --test_sample_num {test_sample_num}'+\
          f' --random_sample_num {random_sample_num}'+\
          f' --num_workers {num_workers}'+\
          f' --test_only'+\
          # f' --verbose'+\
          add_options
          # f' 2>&1 | tee logs/test_{FLAGNAME}'
          )
