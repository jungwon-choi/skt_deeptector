# Copyright © 2020-2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import os
import time
import random
import numpy as np
from collections import OrderedDict
################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
################################################################################
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
################################################################################
from deepfake_dataloader import Deepfake_Dataset
################################################################################
from models.Xception import xception
from models.MesoNet import Meso4, MesoInception4
from models.efficientnet_pytorch.model import EfficientNet
from tools.loss import Weihgted_MSELoss

#===============================================================================
def mean_accuracy(probs, y):
    preds = (probs > 0.5).float()
    return ((preds - y).abs() < 1e-2).float().mean().item()

#===============================================================================
def calc_num_parameters(model):
    return sum([param.nelement() for param in model.parameters()])

#===============================================================================
def set_reproducibility(args):
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

#===============================================================================
def get_model(args):
    if 'Xception'.lower() == args.model.lower():
        model = xception(pretrained=args.pretrained, norm_type=args.norm_type)
        model.fc = nn.Linear(2048, args.num_classes)
    elif 'Meso4'.lower() == args.model.lower():
        model = Meso4(num_classes=args.num_classes)
    elif 'MesoInception4'.lower() == args.model.lower():
        model = MesoInception4(num_classes=args.num_classes)
    elif 'EfficientNet-B'.lower() in args.model.lower():
        model = EfficientNet.get_model(args.model.lower(), args.pretrained, norm_type=args.norm_type)
        feature_dims = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        model._fc = nn.Linear(feature_dims[int(args.model[-1])], args.num_classes)
    elif 'mnasnet0_5' == args.model.lower():
        model = models.mnasnet0_5(pretrained=args.pretrained)
        model.classifier[1] = nn.Linear(1280, args.num_classes)
    elif 'mnasnet0_75' == args.model.lower():
        model = models.mnasnet0_75()
        model.classifier[1] = nn.Linear(1280, args.num_classes)
    elif 'mobilenet_v2' == args.model.lower():
        model = models.mobilenet_v2(pretrained=args.pretrained)
        model.classifier[1] = nn.Linear(1280, args.num_classes)
    elif 'mobilenet_v3_small' == args.model.lower():
        model = models.mobilenet_v3_small(pretrained=args.pretrained)
        model.classifier[3] = nn.Linear(1024, args.num_classes)
    elif 'mobilenet_v3_large' == args.model.lower():
        model = models.mobilenet_v3_large(pretrained=args.pretrained)
        model.classifier[3] = nn.Linear(1024, args.num_classes)
    elif 'VGG16'.lower() == args.model.lower():
        model = models.vgg16_bn(pretrained=args.pretrained)
        model.classifier[6] = nn.Linear(4096, args.num_classes)
    elif 'ResNet18'.lower() == args.model.lower():
        model = models.resnet18(pretrained=args.pretrained)
        model.fc = nn.Linear(512, args.num_classes)
    elif 'ResNet50'.lower() == args.model.lower():
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(2048, args.num_classes)
    elif 'InceptionV3'.lower() == args.model.lower():
        model = models.inception_v3(pretrained=args.pretrained)
        model.fc = nn.Linear(2048, args.num_classes)
    elif 'WRN50'.lower() == args.model.lower():
        model = models.wide_resnet50_2(pretrained=args.pretrained)
        model.fc = nn.Linear(2048, args.num_classes)

    model = model.to(args.device) if args.use_cuda else model

    if args.multi_gpu:
        model = nn.DataParallel(model)
    if args.verbose: print('# of parameters: {}'.format(calc_num_parameters(model)))
    return model

#===============================================================================
def get_transform(args, phase):
    if phase == 'train':
        sequence = []
        # sequence += [transforms.ToPILImage()]
        if args.aug:
            sequence += [transforms.RandomHorizontalFlip(p=0.5)]
            sequence += [transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.88**2, 1.0), ratio=(1., 1.))]
            sequence += [transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.25, contrast=0.1, saturation=0.1, hue=0.1)]), p=0.3)]
            sequence += [transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, sigma=(0.1, 2.0))]), p=0.3)]
            sequence += [transforms.RandomGrayscale(p=0.05)]
            # sequence += [transforms.JpegCompression(p=.2, quality_lower=50),]
        else:
            sequence += [transforms.Resize((args.img_size, args.img_size))] # PIL resize
        sequence += [transforms.ToTensor()]
        sequence += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] # Imagenet
        return transforms.Compose(sequence)
    else:
        sequence = []
        # sequence += [transforms.ToPILImage()]
        sequence += [transforms.Resize((args.img_size, args.img_size))] # PIL resize
        sequence += [transforms.ToTensor()]
        sequence += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] # Imagenet
        return transforms.Compose(sequence)


#===============================================================================
def get_dataset(args, phase='train', domain=None, select_method=None, domain_label=None, real_sample_x=1):
    random.seed(args.seed)

    if phase=='train':
        sample_num=args.train_sample_num
        random_sample_num = None
    elif phase=='val':
        sample_num=args.val_sample_num
        random_sample_num = args.random_sample_num
    elif phase=='test':
        sample_num=args.test_sample_num
        random_sample_num = 10000 #args.random_sample_num

    if domain=='FaceForensics':
        data_path = os.path.join(args.data_root, 'FaceForensics_facecrop')
    elif domain=='CelebDF':
        data_path = os.path.join(args.data_root, 'Celeb-DF-v2_facecrop')
    elif domain=='UADFV':
        data_path = os.path.join(args.data_root, 'UADFV_facecrop')
        # if not args.force_sample:
        sample_num = None
    elif domain=='DeepfakeTIMIT':
        data_path = os.path.join(args.data_root, 'DeepfakeTIMIT_facecrop')
        # if not args.force_sample:
        sample_num = None

    dataset = Deepfake_Dataset(root=data_path,
                        dataset_name=domain,
                        split_type=phase,
                        img_size=args.img_size,
                        transform=get_transform(args, phase),
                        num_classes=args.num_classes,
                        quality=args.quality,
                        sample_num=sample_num,
                        real_sample_x=real_sample_x,
                        auto_stride=True,
                        random_sample_num=random_sample_num,
                        unfair_sample=args.unfair_sample,
                        # with_id=args.with_id,
                        # with_path=args.with_path,
                        # wtih_domain_label=args.wtih_domain_label,
                        domain_label=domain_label,
                        # debug=args.debug,
                        # num_debug_sample=args.num_debug_sample,
                        verbose=args.verbose,
                        # infinite_mode=args.infinite_mode,
<<<<<<< HEAD
                        attack_test=args.attack_test if 'attack_test' in args.__dict__.keys() else False
=======
                        attack_test=args.attack_test
>>>>>>> 03869faef94679d5fb251d72fb9547c716d00e06
                        )
    return dataset

#===============================================================================
def get_envs(args, phase='train'):
    domain_envs = OrderedDict()
    env_domain_list = args.domain_list if phase=='train' else args.total_domain_list

    for d_idx, domain in enumerate(env_domain_list):
        if args.verbose: print('Loading {} dataset'.format(domain))
        domain_envs[domain] = get_dataset(args, phase=phase, domain=domain, domain_label=d_idx)

    if (phase=='train'):# and not args.split_envs): # or args.force_domain
        # Integrate all method into one environment.
        env_list = list(domain_envs.keys())
        for ii in range(len(env_list)-1):
            domain_envs[env_list[0]].all_real_frame_list.extend(domain_envs[env_list[ii+1]].all_real_frame_list)
            domain_envs[env_list[0]].all_fake_frame_list.extend(domain_envs[env_list[ii+1]].all_fake_frame_list)
        domain_envs[env_list[0]].video_frame_list = domain_envs[env_list[0]].all_real_frame_list+domain_envs[env_list[0]].all_fake_frame_list
        args.fake_over_real = float(len(domain_envs[env_list[0]].all_fake_frame_list ) / len(domain_envs[env_list[0]].all_real_frame_list ))

        # if args.unfair_sample:
        #     domain_envs[env_list[0]].reset_pool(fake=2)
        # else:
        #     domain_envs[env_list[0]].reset_pool(fake=0)
        #     domain_envs[env_list[0]].reset_pool(fake=1)
        # if args.verbose: print('pool', [len(domain_envs[env_list[0]].pool[i]) for i in range(3)])
        return {'ERM': domain_envs[env_list[0]]}
    return domain_envs

#===============================================================================
def get_env_dataloaders(envs, train=True, batch_size=32, num_workers=4):
    dataloaders = OrderedDict()
    for env_name in envs.keys():
        dataloaders[env_name] = torch.utils.data.DataLoader(envs[env_name],
                                                batch_size=batch_size,
                                                shuffle=train,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                drop_last=train,)
    return dataloaders

#===============================================================================
def get_criterion(args):
    if args.criterion == 'CE':
        weights = torch.tensor([args.fake_over_real, 1.]) if args.weight_balance else None
        criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weights)
    elif args.criterion == 'BCE':
        pos_weight = torch.tensor([1./args.fake_over_real]) if args.weight_balance else None
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    elif args.criterion == 'MSE':
        pos_weight = 1./args.fake_over_real if args.weight_balance else 1.
        criterion = Weihgted_MSELoss(weight_balance=args.weight_balance, pos_weight=pos_weight)
    return criterion

#===============================================================================
def get_optimizer(args, model):
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay,
                                    )
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                   )
    elif args.optim == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    alpha=0.9,
                                   )
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=args.lr,
                                     betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay,
                                    )
    elif args.optim == 'RAdam':
        raise NotImplementedError

    return optimizer

#===============================================================================
def get_scheduler(args, optimizer):
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    else:
        scheduler = None
    return scheduler

#===============================================================================
def get_roc_graph(labels, preds, fig_label=None, epoch=None):
    matplotlib.use('Agg')
    roc_auc = roc_auc_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds)
    fig = plt.figure(dpi=150)
    plt.plot(fpr, tpr, color='darkorange', label='%s (%0.4f)' % (fig_label, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve'+(' ({})'.format(fig_label) if fig_label else '')+(' @ {} epoch'.format(epoch) if epoch else ''))
    plt.legend(loc="lower right")
    fig.canvas.draw()
    roc_graph = np.transpose(np.array(fig.canvas.renderer._renderer)[...,:3], (2,0,1))
    # For memory saving
    fig.clear()
    plt.close(fig)
    return roc_graph, roc_auc

#===============================================================================
def freeze_layers(args, model):
    for params in model.parameters():
        params.requires_grad = False
    if 'Xception'.lower() == args.model:
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    elif 'EfficientNet'.lower() in args.model:
        model._fc.weight.requires_grad = True
        model._fc.bias.requires_grad = True

#===============================================================================
def unfreeze_layers(args, model):
    if args.model == 'InceptionLSTM':
        for params in model.inceptionv3.parameters():
            params.requires_grad = True
    else:
        for params in model.parameters():
            params.requires_grad = True
