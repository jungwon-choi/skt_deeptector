"""
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from models.batchinstancenorm import BatchInstanceNorm2d
from models.Discriminator import Discriminator

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}

norm_func_dict = {
    'bn': nn.BatchNorm2d,
    'in': nn.InstanceNorm2d,
    'bin': BatchInstanceNorm2d,
    'id': nn.Identity,
}

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False, weight_std=False):
        super(SeparableConv2d,self).__init__()
        Conv2d = ConvWS2d if weight_std else nn.Conv2d

        self.conv1 = Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True, norm_type='bn', weight_std=False):
        super(Block, self).__init__()
        Conv2d = ConvWS2d if weight_std else nn.Conv2d
        Norm2d = norm_func_dict[norm_type]

        if out_filters != in_filters or strides!=1:
            self.skip = Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = Norm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(Norm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(Norm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(Norm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, norm_type='bn', weight_std=False, domain_fc=False, num_domains=4, self_supervised=False):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes
        self.domain_fc = domain_fc
        Conv2d = ConvWS2d if weight_std else nn.Conv2d
        Norm2d = norm_func_dict[norm_type]

        self.conv1 = Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = Norm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(32,64,3,bias=False)
        self.bn2 = Norm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False,norm_type=norm_type,weight_std=weight_std)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1,weight_std=weight_std)
        self.bn3 = Norm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1,weight_std=weight_std)
        self.bn4 = Norm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

        self.self_supervised = self_supervised
        # if self.self_supervised :
        #     self.fc_for_projection = nn.Sequential(
        #         nn.Linear(2048, 2048),
        #         nn.LeakyReLU(),
        #         nn.Linear(2048, 64)
            # )

        if self.domain_fc:
            self.domain_classifier = Discriminator(dims=[2048, 1024, 1024, num_domains], grl=True)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, BatchInstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def extract_features(self, x, get_conv_features=False):
        if get_conv_features: conv_features = []
        x = self.conv1(x)    # 3 -> 32
        x = self.bn1(x)
        x = self.relu1(x)
        # if get_conv_features: conv_features.append(x)

        x = self.conv2(x)   # 32 -> 64
        x = self.bn2(x)
        x = self.relu2(x)
        if get_conv_features: conv_features.append(x)

        x = self.block1(x)  # 64 -> 128
        if get_conv_features: conv_features.append(x)
        x = self.block2(x)  # 128 -> 256
        if get_conv_features: conv_features.append(x)
        x = self.block3(x)  # 256 -> 728
        if get_conv_features: conv_features.append(x)
        if get_conv_features: return conv_features
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x) # 728 -> 728
        x = self.block12(x) # 728 -> 1024

        x = self.conv3(x)   # 1024 -> 1536
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)   # 1536 -> 2048
        x = self.bn4(x)
        x = self.relu4(x)

        return x

    def forward(self, x, get_feature=False):
        x = self.extract_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if get_feature:
            feats = x.clone()
            x = self.fc(x)
            return x, feats
        if self.domain_fc:
            feats = x.clone()
            class_logits = self.fc(x)
            domain_logits = self.domain_classifier(feats)
            return class_logits, domain_logits
        # if self.self_supervised :
        #     detect = self.fc(x)
        #     contrast = self.fc_for_projection(x)
        #     return detect, contrast
        x = self.fc(x)
        return x


def xception(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']), strict=False)
    return model

class Xception_Shallow(nn.Module):
    def __init__(self, num_classes=1000, norm_type='bn', weight_std=False, domain_fc=False, num_domains=4, self_supervised=False):
        super(Xception_Shallow, self).__init__()

        self.num_classes = num_classes
        self.domain_fc = domain_fc
        Conv2d = ConvWS2d if weight_std else nn.Conv2d
        Norm2d = norm_func_dict[norm_type]

        self.conv1 = Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = Norm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(32,64,3,bias=False)
        self.bn2 = Norm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True,norm_type=norm_type,weight_std=weight_std)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False,norm_type=norm_type,weight_std=weight_std)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1,weight_std=weight_std)
        self.bn3 = Norm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1,weight_std=weight_std)
        self.bn4 = Norm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

        # self.self_supervised = self_supervised
        # if self.self_supervised :
        #     self.fc_for_projection = nn.Sequential(
        #         nn.Linear(2048, 2048),
        #         nn.LeakyReLU(),
        #         nn.Linear(2048, 64)
        #     )

        if self.domain_fc:
            self.domain_classifier = Discriminator(dims=[2048, 1024, 1024, num_domains], grl=True)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, BatchInstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def extract_features(self, x, get_conv_features=False):
        if get_conv_features: conv_features = []
        x = self.conv1(x)    # 3 -> 32
        x = self.bn1(x)
        x = self.relu1(x)
        # if get_conv_features: conv_features.append(x)

        x = self.conv2(x)   # 32 -> 64
        x = self.bn2(x)
        x = self.relu2(x)
        if get_conv_features: conv_features.append(x)

        x = self.block1(x)  # 64 -> 128
        if get_conv_features: conv_features.append(x)
        x = self.block2(x)  # 128 -> 256
        if get_conv_features: conv_features.append(x)
        x = self.block3(x)  # 256 -> 728
        if get_conv_features: conv_features.append(x)
        if get_conv_features: return conv_features
        x = self.block12(x) # 728 -> 1024

        x = self.conv3(x)   # 1024 -> 1536
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)   # 1536 -> 2048
        x = self.bn4(x)
        x = self.relu4(x)

        return x

    def forward(self, x, get_feature=False):
        x = self.extract_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if get_feature:
            feats = x.clone()
            x = self.fc(x)
            return x, feats
        if self.domain_fc:
            feats = x.clone()
            class_logits = self.fc(x)
            domain_logits = self.domain_classifier(feats)
            return class_logits, domain_logits
        # if self.self_supervised :
        #     detect = self.fc(x)
        #     contrast = self.fc_for_projection(x)
        #     return detect, contrast
        x = self.fc(x)
        return x


def xception_shallow(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception_Shallow(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']), strict=False)
    return model
