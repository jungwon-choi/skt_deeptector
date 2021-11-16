# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
import argparse
import time
import os

from tools.utils import printflush as print
from tools.utils import *
from tools.iterator import Iterator
from tools.args import TestArgs


from models.Xception import xception

#===============================================================================
parser = argparse.ArgumentParser(description='Deepfake Detector test code')
args = TestArgs().initialize(parser)
#===============================================================================

def main(args):
    # Fix the seed for reproducibility
    set_reproducibility(args)

    # Build the model
    if not args.ensemble:
        model = get_model(args)
        # print(model)
        checkpoint_dict = torch.load(args.ckpt_path, map_location=args.device)

        if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint_dict['model_state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            checkpoint_dict['model_state_dict'] = new_state_dict

        model.load_state_dict(checkpoint_dict['model_state_dict'])
    else:
        models = []
        for ckpt_path, ckpt_opt_list in zip(args.ensemble_ckpt_list, args.ensemble_opt_list):

            reset_option(args, ckpt_opt_list)
            model = get_model(args)

            checkpoint_dict = torch.load(ckpt_path, map_location=args.device)

            if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint_dict['model_state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                checkpoint_dict['model_state_dict'] = new_state_dict

            model.load_state_dict(checkpoint_dict['model_state_dict'])
            models.append(model)
        model = models

    # print('[Notice] Load best score model: {} epochs ({} iters)'.format(checkpoint_dict['current_epoch'],
    #                                                                     checkpoint_dict['current_iter']))
    if 'best_threshold' in checkpoint_dict:
        best_threshold = checkpoint_dict['best_threshold'] # 0.5
        print('[Notice] best treshold: {:.3f}'.format(best_threshold))
    else:
        best_threshold = 0.5
    print('[Notice] Model ready.')

    # Set environments dataLoaders
    envs = {}
    dataloaders = {}
    phase = 'test'
    envs[phase] = get_envs(args, phase=phase, attack=True)
    dataloaders[phase] = get_env_dataloaders(args, envs[phase], train=False,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    for env_name in dataloaders[phase].keys():
        dataset = dataloaders[phase][env_name].dataset
        num_real_framse = len(dataset.all_real_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_real_frame)
        num_fake_framse = len(dataset.all_fake_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_fake_frame)
        print('{:<6} {:<15}: real {:5d}, fake {:5d}'.format(phase, env_name, num_real_framse, num_fake_framse))
    print('[Notice] Dataloader ready.')

    # Set iterator
    iterator = Iterator(args, model, dataloaders, best_threshold=best_threshold)
    print('[Notice] Iterator ready.')

    # base model to attack
    if args.attack_base_model == 'Xception':
        base_model = xception(pretrained=args.pretrained, norm_type=args.norm_type)
        base_model.fc = nn.Linear(2048, args.num_classes)
        base_model_ckpt_path = '../../individual_research/deepfake_balhae/checkpoints/210610_125553_test/best_ckpt'

    elif args.attack_base_model == 'mb3l':
        import torchvision.models as models
        base_model = models.mobilenet_v3_large(pretrained=args.pretrained)
        base_model.classifier[3] = nn.Linear(1280, args.num_classes)
        base_model_ckpt_path = 'checkpoints/211029_121429_base_mb3l/best_ckpt'

    elif args.attack_base_model == 'mb3s':
        import torchvision.models as models
        base_model = models.mobilenet_v3_small(pretrained=args.pretrained)
        base_model.classifier[3] = nn.Linear(1024, args.num_classes)
        base_model_ckpt_path = 'checkpoints/211029_121423_base_mb3s/best_ckpt'
    
    elif args.attack_base_model == 'efficientnet-b4':
        from models.efficientnet_pytorch.model import EfficientNet
        base_model = EfficientNet.get_model('EfficientNet-B4'.lower(), args.pretrained)
        feature_dims = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        base_model._fc = nn.Linear(feature_dims[int(args.model[-1])], args.num_classes)
        base_model_ckpt_path = 'checkpoints/211029_121415_base_efb4/best_ckpt'

    base_model_ckpt = torch.load(base_model_ckpt_path, map_location=args.device)
    base_model.load_state_dict(base_model_ckpt['model_state_dict'])
    print('[Notice] Load base model to attack :'+args.attack_base_model+' {} epochs ({} iters)'.format(checkpoint_dict['current_epoch'],
                                                                            checkpoint_dict['current_iter']))
    base_model.cuda()
    base_model.eval()


    # Test the model
    test_start = time.time()
    #=======================================================================
    # iterator.test()
    # eps = args.adversarial_attack_eps
    iterator.attack_test(base_model)
    #=======================================================================
    test_end = time.time()
    print("[Notice] Test elapsed time: {0:.3f} sec".format(test_end - test_start))

    # Test Inference time
    # dummy_input = torch.randn(1, 3,args.img_size,args.img_size, dtype=torch.float).to(args.device)
    #
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 1000
    # timings=np.zeros((repetitions,1))
    # # GPU-WARM-UP
    # for _ in range(10):
    #     _ = model(dummy_input)
    # # MEASprintlushRFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model(dummy_input)
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(mean_syn, '+-', std_syn)
    # exit()

#===============================================================================
if __name__ == '__main__':
    print(f"[Start time] {time.strftime('%c', time.localtime(time.time()))}")
    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("\n[Notice] Total elapsed time: {0:.3f} sec".format(end - start))
    print(f"[Complete time] {time.strftime('%c', time.localtime(time.time()))}", end='\n'*2)
