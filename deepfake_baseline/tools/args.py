# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import os
import time
import torch

class Args():
    def __init__(self):
        self.initialized = False
    ############################################################################
    def initialize(self, parser):
        if self.initialized:
            return self.args
        # Model
        parser.add_argument('--model', type=str, default='Xception',
                            help='Select model to train among [Xception|Meso4|MesoInception4|EfficientNet-B0~8|VGG16|ResNet50|ResNeXt101|WRN50]')
        parser.add_argument('--pretrained', action='store_true',
                            help='Load model weight pretrained form Imagenet')
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--norm_type', type=str, default='bn',
                            help='[bn|in|bin]')

        # Dataset
        parser.add_argument('--dataset', type=str, default=None,
                            help='Select domains. [FaceForensics|UADFV|DeepfakeTIMIT|CelebDF]')
        parser.add_argument('--quality', type=float, default=100,
                            help='Image quality (JEPG compress) [default=100]')
        parser.add_argument('--img_size', type=int, default=299,
                            help='Input image size.')
        parser.add_argument('--train_sample_num', type=int, default=40,
                            help='the number of frames per video for training')
        parser.add_argument('--val_sample_num', type=int, default=40,
                            help='the number of frames per video for validation')
        parser.add_argument('--test_sample_num', type=int, default=100,
                            help='the number of frames per video for test')
        parser.add_argument('--random_sample_num', type=int, default=5000,
                            help='the number of frames for fair evaluation')
        parser.add_argument('--split_envs', action='store_true')
        parser.add_argument('--unfair_sample', action='store_true')
        parser.add_argument('--aug', action='store_true')
        parser.add_argument('--num_workers', type=int, default=8)

        # Optimize
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--max_iters', type=int, default=None)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--criterion', type=str, default='CE',
                            help='Select classifier loss. [BCE|CE|MSE]')
        parser.add_argument('--weight_balance', action='store_true',
                            help='add weight for insufficient real data')
        parser.add_argument('--optim', type=str, default='Adam',
                            help='Select optimizer. [Adam|SGD|AdamW|RAdam]')
        parser.add_argument('--scheduler', type=str, default=None,
                            help='Select scheduler. [StepLR|CosineAnnealing]')
        parser.add_argument('--lr_decay_step', type=int, default=1000)
        parser.add_argument('--lr_decay_rate', type=float, default=0.98)
        parser.add_argument('--earlystop', action='store_true')

        # Others
        parser.add_argument('--data_root', type=str, default='/SSD/dfdc/datasets')
        parser.add_argument('--ckpt_path', type=str, default='./checkpoints/')
        parser.add_argument('--logs_path', type=str, default='./logs/')
        parser.add_argument('--print_freq', type=int, default=20)
        parser.add_argument('--val_freq', type=int, default=2000)
        parser.add_argument('--gpu_idx', type=str, default=None)
        parser.add_argument('--norecord', action='store_true')
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--flag', type=str, default=None)
        parser.add_argument('--verbose', action='store_true')
        args = parser.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_idx
        args.FILE_FORMAT = time.strftime('%y%m%d_%H%M%S_', time.localtime(time.time()))+args.flag
        args.CHECKPOINT_DIR = os.path.join(args.ckpt_path, args.FILE_FORMAT)
        args.ARGS_INFO_PATH = os.path.join(args.ckpt_path, args.FILE_FORMAT, 'args_info.txt')
        args.LAST_CHECKPOINT_PATH  = os.path.join(args.ckpt_path, args.FILE_FORMAT, 'last_ckpt')
        args.BEST_CHECKPOINT_PATH  = os.path.join(args.ckpt_path, args.FILE_FORMAT, 'best_ckpt')
        args.FINAL_CHECKPOINT_PATH = os.path.join(args.ckpt_path, args.FILE_FORMAT, 'final_ckpt')
        args.domain_list = args.dataset.split(',')
        args.total_domain_list = ['FaceForensics', 'UADFV', 'DeepfakeTIMIT', 'CelebDF']
        args.use_cuda = torch.cuda.is_available()
        args.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
        args.num_gpus = torch.cuda.device_count()
        args.multi_gpu = args.num_gpus > 1
        args.local_rank = 0

        self.args = args
        self.initialized = True

        self.print_args()
        return self.args
    ############################################################################
    def print_args(self):
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{:35s}\t: {:}".format(arg, content))
        print("==========     CONFIG END    =============")

class TestArgs():
    def __init__(self):
        self.initialized = False
    ############################################################################
    def initialize(self, parser):
        if self.initialized:
            return self.args
        # Model
        parser.add_argument('--model', type=str, default='Xception',
                            help='Select model to train among [Xception|Meso4|MesoInception4|EfficientNet-B0~8|VGG16|ResNet50|ResNeXt101|WRN50]')
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--pretrained', action='store_true',
                            help='Load model weight pretrained form Imagenet')
        parser.add_argument('--norm_type', type=str, default='bn',
                            help='[bn|in|bin]')

        # Dataset
        parser.add_argument('--dataset', type=str, default=None,
                            help='Select domains. [FaceForensics|UADFV|DeepfakeTIMIT|CelebDF]')
        parser.add_argument('--quality', type=float, default=100,
                            help='Image quality (JEPG compress) [default=100]')
        parser.add_argument('--img_size', type=int, default=299,
                            help='Input image size.')
        parser.add_argument('--val_sample_num', type=int, default=40,
                            help='the number of frames per video for validation')
        parser.add_argument('--test_sample_num', type=int, default=100,
                            help='the number of frames per video for test')
        parser.add_argument('--random_sample_num', type=int, default=5000,
                            help='the number of frames for fair evaluation')
        parser.add_argument('--split_envs', action='store_true')
        parser.add_argument('--unfair_sample', action='store_true')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=128)

        # Others
        parser.add_argument('--data_root', type=str, default='/SSD/dfdc/datasets')
        parser.add_argument('--ckpt_path', type=str, default='./checkpoints/')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--gpu_idx', type=str, default=None)
        parser.add_argument('--attack_test', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        args = parser.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_idx
        args.domain_list = args.dataset.split(',')
        args.total_domain_list = ['FaceForensics', 'UADFV', 'DeepfakeTIMIT', 'CelebDF']
        args.use_cuda = torch.cuda.is_available()
        args.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
        args.num_gpus = torch.cuda.device_count()
        args.multi_gpu = args.num_gpus > 1
        self.args = args
        self.initialized = True

        self.print_args()
        return self.args

    ############################################################################
    def print_args(self):
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{:35s}\t: {:}".format(arg, content))
        print("==========     CONFIG END    =============")
