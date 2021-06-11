# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
import argparse
import time
import os

from tools.utils import *
from tools.iterator import Iterator
from tools.args import TestArgs

import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

#===============================================================================
parser = argparse.ArgumentParser(description='Deepfake Detector test adversarial attack code')
args = TestArgs().initialize(parser)
#===============================================================================

def main(args):
    # Fix the seed for reproducibility
    set_reproducibility(args)

    # Build the model
    model = get_model(args).eval()
    checkpoint_dict = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    print('[Notice] Load best score model: {} epochs ({} iters)'.format(checkpoint_dict['current_epoch'],
                                                                            checkpoint_dict['current_iter']))
    fmodel = PyTorchModel(model, bounds=(0, 1))

    # model = models.resnet18(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    print('[Notice] Model ready.')

    # Set environments dataLoaders
    envs = {}
    dataloaders = {}
    phase = 'test'
    envs[phase] = get_envs(args, phase=phase)
    dataloaders[phase] = get_env_dataloaders(envs[phase], train=False,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    # for env_name in dataloaders[phase].keys():
    #     dataset = dataloaders[phase][env_name].dataset
    #     num_real_framse = len(dataset.all_real_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_real_frame)
    #     num_fake_framse = len(dataset.all_fake_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_fake_frame)
    #     print('{:<6} {:<15}: real {:5d}, fake {:5d}'.format(phase, env_name, num_real_framse, num_fake_framse))
    print('[Notice] Dataloader ready.')

    # Test adversarial attack
    # images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    images, labels = iter(dataloaders[phase]['FaceForensics']).next()
    images = images.to(args.device)
    labels = labels.squeeze(dim=1).to(args.device)
    print('labels:', labels.tolist())

    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    attack = LinfPGD()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    robust_accuracy = 1 - success.float().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


#===============================================================================
if __name__ == '__main__':
    print("[Start time]", time.strftime('%c', time.localtime(time.time())))
    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("\n[Notice] Total elapsed time: {0:.3f} sec".format(end - start))
    print("[Complete time]", time.strftime('%c', time.localtime(time.time())), end='\n'*2)
