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
from models.Xception import xception, xception_shallow
from models.MesoNet import Meso4, MesoInception4
from models.efficientnet_pytorch.model import EfficientNet
from models.oudefend import OUDefend
from models.simclr import fc_layer
from tools.loss import Weihgted_MSELoss, contrastive_loss
# from tools.parallel import DataParallelModel, DataParallelCriterion

################################################################################
import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img

#===============================================================================
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, args, base_transform, n_views=2, aug_hard=False):
        if aug_hard :
            sequence = []
            sequence += [transforms.Resize((args.img_size, args.img_size))] # PIL resize
            sequence += [transforms.ToTensor()]
            sequence += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] # Imagenet
            self.base_transform =  transforms.Compose(sequence)
        else :
            self.base_transform = base_transform
        self.n_views = n_views
        self.aug_hard = aug_hard
        self.args = args
    def __call__(self, x):
        if self.aug_hard :
            self.base_transform.transforms.insert(0, RandAugment(self.args.N, self.args.M))
        return [self.base_transform(x) for i in range(self.n_views)]

#===============================================================================
def reset_option(args, ckpt_opt_list):
    ckpt_opt_list = ckpt_opt_list.split('/')
    print('-'*30)
    for ckpt_opt in ckpt_opt_list:
        attr, value = ckpt_opt.split(':')
        attr, value = attr.strip(), value.strip()
        if value in ['True', 'False']:
            value = False if value == 'False' else True
        # else:
        #     value = float(value)
        print(f'{attr}={value}')
        setattr(args, attr, value)

#===============================================================================
def printflush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

#===============================================================================
def mean_accuracy(probs, y, threshold=0.5):
    preds = (probs > threshold).float()
    return ((preds - y).abs() < 1e-2).float().mean().item()

#===============================================================================
def calc_num_parameters(model):
    return sum([param.nelement() for param in model.parameters()])

#===============================================================================
def set_reproducibility(args):
    if args.seed == -1:
        pass
    else:
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
    num_channel = 3
    if args.phase_concat or args.phase_only:
        num_channel = 4 if args.phase_concat else 1

    if 'Xception'.lower() == args.model.lower():
        model = xception(pretrained=args.pretrained, norm_type=args.norm_type, self_supervised=args.self_supervised)
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # Adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        model.fc = fc_layer(2048, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(2048, args.num_classes)

    elif 'Xception_oudefend'.lower() == args.model.lower():
        model = xception(pretrained=args.pretrained, norm_type=args.norm_type, self_supervised=args.self_supervised)
        # phase channel concatenate part
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # Adversarial attack robustness part
        oudefend = OUDefend(num_channel)
        model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        model.fc = fc_layer(2048, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(2048, args.num_classes)

    elif 'Xception_Shallow'.lower() == args.model.lower():
        model = xception_shallow(pretrained=args.pretrained, norm_type=args.norm_type, self_supervised=args.self_supervised)
        # phase channel concatenate part
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        model.fc = fc_layer(2048, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(2048, args.num_classes)
    elif 'Meso4'.lower() == args.model.lower():
        model = Meso4(num_classes=args.num_classes, num_feature=args.num_feature)
        # phase channel concatenate part
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        if args.self_supervised:
            model.fc = fc_layer(args.num_feature, args.num_classes)

    elif 'MesoInception4'.lower() == args.model.lower():
        model = MesoInception4(num_channel=num_channel, num_classes=args.num_classes, oudefend=args.oudefend, num_feature=args.num_feature)
        # self-supervised learning part
        if args.self_supervised:
            model.fc = fc_layer(args.num_feature, args.num_classes)

    elif 'EfficientNet-B'.lower() in args.model.lower():
        model = EfficientNet.get_model(args.model.lower(), args.pretrained, in_channels=num_channel, num_classes=args.num_classes, norm_type=args.norm_type, oudefend=args.oudefend)
        feature_dims = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        # model._fc = nn.Linear(feature_dims[int(args.model[-1])], args.num_classes)
        # self-supervised learning part
        if args.self_supervised:
            model._fc = fc_layer(feature_dims[int(args.model[-1])], args.num_classes)

    elif 'mobilenet_v2' == args.model.lower():
        model = models.mobilenet_v2(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.features[0][0] = nn.Conv2d(num_channel, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.features[0][0] = nn.Sequential(oudefend, model.features[0][0])
        # self-supervised learning part
        model.classifier[1] = fc_layer(1280, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(1280, args.num_classes)

    elif 'mobilenet_v3_small' == args.model.lower():
        model = models.mobilenet_v3_small(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.features[0][0] = nn.Conv2d(num_channel, 16, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.features[0][0] = nn.Sequential(oudefend, model.features[0][0])
        # self-supervised learning part
        model.classifier[3] = fc_layer(1024, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(1024, args.num_classes)

    elif 'mobilenet_v3_large' == args.model.lower():
        model = models.mobilenet_v3_large(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.features[0][0] = nn.Conv2d(num_channel, 16, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.features[0][0] = nn.Sequential(oudefend, model.features[0][0])
        # self-supervised learning part
        model.classifier[3] = fc_layer(1280, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(1280, args.num_classes)

    elif 'VGG16'.lower() == args.model.lower():
        model = models.vgg16_bn(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.features[0] = nn.Conv2d(num_channel, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.features[0] = nn.Sequential(oudefend, model.features[0])
        # self-supervised learning part
        model.classifier[6] = fc_layer(4096, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(4096, args.num_classes)

    elif 'ResNet18'.lower() == args.model.lower():
        model = models.resnet18(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        model.fc = fc_layer(512, args.num_classes) if args.self_supervised \
                                                   else nn.Linear(512, args.num_classes)

    elif 'ResNet50'.lower() == args.model.lower():
        model = models.resnet50(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        model.fc = fc_layer(2048, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(2048, args.num_classes)

    elif 'WRN50'.lower() == args.model.lower():
        model = models.wide_resnet50_2(pretrained=args.pretrained)
        # phase channel concatenate part
        if num_channel != 3:
            model.conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
        # adversarial attack robustness part
        if args.oudefend:
            oudefend = OUDefend(num_channel)
            model.conv1 = nn.Sequential(oudefend, model.conv1)
        # self-supervised learning part
        model.fc = fc_layer(2048, args.num_classes) if args.self_supervised \
                                                    else nn.Linear(2048, args.num_classes)
    else:
        print(args.model, 'is not implemented! Please check the model name.')
        exit()

    if args.inference_test and args.self_supervised:
        if 'EfficientNet-B'.lower() in args.model.lower():
            del model._fc.projection_layer
            model._fc.projection_layer = None
        elif 'mobilenet_v3' in args.model.lower():
            del model.classifier[3].projection_layer
            model.classifier[3].projection_layer = None

    model = model.to(args.device) if args.use_cuda else model

    if args.multi_gpu:
        model = nn.DataParallel(model)
        # model = DataParallelModel(model)
        # model = nn.parallel.DistributedDataParallel(model,
        #                                             device_ids=[args.local_rank],
        #                                             output_device=args.local_rank)
    if args.verbose: printflush('# of parameters: {}'.format(calc_num_parameters(model)))
    return model

#===============================================================================
def get_oudefend_output(args, model, inputs):
    if args.multi_gpu:
        model = model.module
    if args.model.lower() in ['xception', 'xception_oudefend', 'xception_shallow',
                              'meso4','resnet18', 'resnet50', 'wrn50']:
        oudefend_output = model.conv1[0](inputs)
    elif args.model.lower() in ['inceptionv3']:
        oudefend_output = model.Conv2d_1a_3x3.conv[0](inputs)
    elif args.model.lower() in ['vgg16']:
        oudefend_output = model.features[0][0](inputs)
    elif args.model.lower() in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
        oudefend_output = model.features[0][0][0](inputs)
    elif args.model.lower() in ['mesoinception4']:
        oudefend_output = model.oudefend_net(inputs)
    elif 'EfficientNet-B'.lower() in args.model.lower():
        oudefend_output = model.oudefend_net(inputs)
    return oudefend_output

#===============================================================================
def get_classification_params(args, model):
    if args.multi_gpu:
        model = model.module
    if args.model.lower() in ['xception', 'xception_oudefend', 'xception_shallow',
                              'meso4', 'mesoinception4',
                              'resnet18', 'resnet50', 'inceptionv3', 'wrn50']:

        #params_ = model.fc.classification_layer.parameters()
        #params_add =  model.fc.fc_layer_.parameters()
        #params = list(params_) + list(params_add)
        params = model.fc.classification_layer.parameters()

    elif args.model.lower() in ['mobilenet_v3_small', 'mobilenet_v3_large']:
        params = model.classifier[3].classification_layer.parameters()
    elif args.model.lower() in ['vgg16']:
        params = model.classifier[6].classification_layer.parameters()
    elif args.model.lower() in ['mobilenet_v2']:
        params = model.classifier[1].classification_layer.parameters()
    elif 'EfficientNet-B'.lower() in args.model.lower():
        params = model._fc.classification_layer.parameters()
    return params

#===============================================================================
def get_transform(args, phase):
    if phase == 'train' or args.env_shift_test:
        sequence = []
        # sequence += [transforms.ToPILImage()]
        if args.env_shift_test or args.aug:
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
def get_dataset(args, phase='train', domain=None, select_method=None, domain_label=None, real_sample_x=1, attack=False):
    random.seed(args.seed)

    if phase=='train':
        sample_num=args.train_sample_num
        random_sample_num = None
    elif phase=='val':
        sample_num=args.val_sample_num
        random_sample_num = args.val_random_sample_num
    elif phase=='test':
        sample_num=args.test_sample_num
        random_sample_num = args.random_sample_num

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
                        transform= (ContrastiveLearningViewGenerator(args,
                                    get_transform(args, phase),
                                    args.n_views, aug_hard=True if (not args.pretrain_already) else False)) \
                                    if (args.self_supervised and phase == 'train') \
                                    else get_transform(args, phase),
                        num_classes=args.num_classes,
                        quality=args.quality,
                        sample_num=sample_num,
                        real_sample_x=real_sample_x,
                        auto_stride=True,
                        random_sample_num=random_sample_num,
                        unfair_sample=args.unfair_sample,
                        domain_label=domain_label,
                        verbose=args.verbose,
                        attack_test=args.attack_test if 'attack_test' in args.__dict__.keys() else False,
                        phase_concat=args.phase_concat,
                        phase_only=args.phase_only,
                        attack=attack,
                        )
    return dataset

#===============================================================================
def get_envs(args, phase='train', attack=False):
    domain_envs = OrderedDict()
    env_domain_list = args.domain_list if phase=='train' else args.total_domain_list

    for d_idx, domain in enumerate(env_domain_list):
        if args.verbose: printflush('Loading {} dataset'.format(domain))
        domain_envs[domain] = get_dataset(args, phase=phase, domain=domain, domain_label=d_idx, attack=attack)

    if (phase=='train'):# and not args.split_envs): # or args.force_domain
        # Integrate all method into one environment.
        env_list = list(domain_envs.keys())
        for ii in range(len(env_list)-1):
            domain_envs[env_list[0]].all_real_frame_list.extend(domain_envs[env_list[ii+1]].all_real_frame_list)
            domain_envs[env_list[0]].all_fake_frame_list.extend(domain_envs[env_list[ii+1]].all_fake_frame_list)
        domain_envs[env_list[0]].video_frame_list = domain_envs[env_list[0]].all_real_frame_list+domain_envs[env_list[0]].all_fake_frame_list
        args.fake_over_real = float(len(domain_envs[env_list[0]].all_fake_frame_list ) / len(domain_envs[env_list[0]].all_real_frame_list ))

        return {'ERM': domain_envs[env_list[0]]}
    return domain_envs

#===============================================================================
def get_env_dataloaders(args, envs, train=True, batch_size=32, num_workers=4):
    dataloaders = OrderedDict()
    for env_name in envs.keys():
        sampler = None #if not args.multi_gpu \
                       #else torch.utils.data.distributed.DistributedSampler(envs[env_name])
        dataloaders[env_name] = torch.utils.data.DataLoader(envs[env_name],
                                                batch_size=batch_size,
                                                shuffle=True,#train,
                                                num_workers=num_workers,
                                                pin_memory=False,
                                                drop_last=train,
                                                sampler=sampler,
                                                #persistent_workers=True,
                                                )
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
    criterion = criterion.to(args.device)
    # if args.multi_gpu:
    #     criterion = DataParallelCriterion(criterion)
    return criterion

#===============================================================================
def get_optimizer(args, model, params=None):
    if params is None:
        params = model.parameters()
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(params,
                                     lr=args.lr,
                                     betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay,
                                    )
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                   )
    elif args.optim == 'RMSProp':
        optimizer = torch.optim.RMSprop(params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    alpha=0.9,
                                   )
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(params,
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
