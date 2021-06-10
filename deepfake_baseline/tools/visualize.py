# Copyright © 2020-2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

def normalize_grad(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = gradient.transpose(1, 2, 0)
    gradient = np.uint8(gradient*255)
    return gradient

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
#     0.299R + 0.587G + 0.114B
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
#     grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return np.uint8(grayscale_im*255)

def get_guided_gradcam(heatmap, grad):
    if len(grad.shape) > 2:
        gd_cam = np.uint8(np.multiply(heatmap[:,:,None], grad))
    else:
        gd_cam = np.uint8(np.multiply(heatmap, grad))
    return gd_cam

def get_blended_result(image, heatmap, alpha=0.5, style=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(heatmap, style)[:,:,::-1]
    if image.size[0]!=heatmap.shape[1] or image.size[1]!=heatmap.shape[0]:
        image = T.Resize((heatmap.shape[0],heatmap.shape[1]))(image)
    img = np.uint8(np.array(image))
    superimposed_img = (heatmap * alpha + img)/(1+alpha)
    return Image.fromarray(np.uint8(superimposed_img))

class GradCAM():
    def __init__(self, model):
        # get the pretrained model
        self.model = model
        self.model.eval()
        # placeholder for the gradients
        self.gradients = None

    def get_result(self, image, class_idx=1):
        device = next(self.model.parameters()).device
        image = image.to(device)
        pred = self.forward(image)
        # get the gradient of the output with respect to the parameters of the model
        pred[:, class_idx].backward()
        # pull the gradients out of the model
        gradients = self.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # get the activations of the last convolutional layer
        activations = self.get_activations(image).detach()
        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        if 'cuda' in str(device):
            heatmap = heatmap.cpu()

        # relu on top of the heatmap
        heatmap = np.maximum(heatmap, 0)
        # normalize the heatmap
        heatmap = heatmap - heatmap.min()
        heatmap /= heatmap.max()
        heatmap = np.uint8(heatmap*255)
        heatmap = np.uint8(Image.fromarray(heatmap).resize((image.shape[2], image.shape[3]), Image.ANTIALIAS))/255
        return heatmap

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.model.extract_features(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.model.extract_features(x)


class GuidedBackprop():
    def __init__(self, model):
        # get the pretrained model
        self.model = model
        self.model.eval()
        # placeholder for the gradients
        self.gradients = None
        self.forward_relu_outputs = []

        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]#self.model.conv1
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)
        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        for pos, module in self.model.named_modules():
            if 'Xception' in  str(type(module)): continue
            if type(module) == nn.Sequential: continue
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def get_result(self, image, class_idx=1):
        device = next(self.model.parameters()).device
        image = image.to(device)
        image.requires_grad_()
        # Forward pass
        pred = self.forward(image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, pred.size()[-1]).zero_()
        one_hot_output[0][class_idx] = 1
        if 'cuda' in str(device):
            one_hot_output = one_hot_output.to(device)
        # Backward pass
        pred.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        if 'cuda' in str(device):
            gradient = self.gradients.data.cpu().numpy()[0]
        else:
            gradient = self.gradients.data.numpy()[0]
        pos_saliency = (np.maximum(0, gradient) / gradient.max())
        neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
        return gradient, pos_saliency, neg_saliency

    def forward(self, x):
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return x

class GuidedGradCAM():
    def __init__(self, model):
        self.gradcam = GradCAM(model)
        self.guidedbackprop = GuidedBackprop(model)

    def get_result(self, image, class_idx=1):
        heatmap = self.gradcam.get_result(image, class_idx)
        both_sal, pos_sal, neg_sal = self.guidedbackprop.get_result(image, class_idx)
        vis_both = convert_to_grayscale(both_sal)
        vis_pos = convert_to_grayscale(pos_sal)
        vis_neg = convert_to_grayscale(neg_sal)

        result = {
            'gradcam': np.uint8(heatmap*255),
            'guidedbp': {
                'both': vis_both,
                'pos': vis_pos,
                'neg': vis_neg,
            },
            'guidedgradcam': {
                'both': get_guided_gradcam(heatmap, vis_both),
                'pos': get_guided_gradcam(heatmap, vis_pos),
                'neg': get_guided_gradcam(heatmap, vis_neg),
            }
        }
        return result
