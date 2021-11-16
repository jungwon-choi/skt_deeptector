# Copyright © 2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
import argparse
import random
import time
import os

from tools.utils import printflush as print
from tools.utils import *
from tools.iterator import Iterator
from tools.args import Args


#===============================================================================
parser = argparse.ArgumentParser(description='Deepfake Detector train code')
args = Args().initialize(parser)
#===============================================================================
def main(args):
    # Fix the seed for reproducibility
    set_reproducibility(args)

    # Build the model
    model = get_model(args)
    print('[Notice] Model ready.\n')

    # Build Environment dataLoaders
    envs = {}
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        phase_batch_size = int(args.batch_size/len(args.domain_list)) if args.split_envs and phase=='train' else args.batch_size
        print('Get {:<5} dataloader '.format(phase)+'-'*26)
        envs[phase] = get_envs(args, phase=phase)
        dataloaders[phase] = get_env_dataloaders(args, envs[phase], train=(phase=='train'),
                                                batch_size=phase_batch_size,
                                                num_workers=args.num_workers)
        for env_name in dataloaders[phase].keys():
            dataset = dataloaders[phase][env_name].dataset
            num_real_framse = len(dataset.all_real_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_real_frame)
            num_fake_framse = len(dataset.all_fake_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_fake_frame)
            print('{:<6} {:<15}: real {:5d}, fake {:5d}'.format(phase, env_name, num_real_framse, num_fake_framse))
    print('[Notice] Dataloader ready.\n')


    if args.self_supervised and args.pretrain_epochs > 0:
        if args.pretrain_already :
            #checkpoint_dict = torch.load('./checkpoints/211101_182849_randaug_b4_N2M28B64/pretrain_ckpt_2000', map_location=args.device)
            #checkpoint_dict = torch.load('./checkpoints/211101_223159_randaug_b4_N2M28B32/pretrain_ckpt', map_location=args.device)
            checkpoint_dict = torch.load('./checkpoints/211102_110023_randaug_b4_N10M28B32/pretrain_ckpt', map_location=args.device)

            #checkpoint_dict = torch.load('./checkpoints/211101_205020_randaug_mobile_N2M10B64/pretrain_ckpt', map_location=args.device)
            #checkpoint_dict = torch.load('./checkpoints/211101_220304_randaug_mobile_N2M28B32/pretrain_ckpt', map_location=args.device)
            #checkpoint_dict = torch.load('./checkpoints/211102_125958_randaug_mbl_N10M28B32/pretrain_ckpt', map_location=args.device)
            #checkpoint_dict = torch.load('./checkpoints/211102_145343_randaug_mbl_N10M28B32_ouF/pretrain_ckpt', map_location=args.device)
            #checkpoint_dict = torch.load('./checkpoints/211103_113604_randaug_mbs_N10M28B32/pretrain_ckpt', map_location=args.device)
            #checkpoint_dict = torch.load('./checkpoints/211103_175247_randaug_mbs_N10M28B32_ouF/pretrain_ckpt', map_location=args.device)
            # checkpoint_dict = torch.load('./checkpoints/211103_185617_randaug_mbs_N10M28B32_ouF_ouP/pretrain_ckpt', map_location=args.device)


            pretrained_dict = {}
            for key in checkpoint_dict.keys() :
                val = checkpoint_dict[key]
                pretrained_dict[key[7:]] = val
            #model.module.load_state_dict(dict_)
            if args.multi_gpu :
                new_model_dict = model.module.state_dict()
            else :
                new_model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
            #del pretrained_dict['classifier[3].classification_layer.2.weight']
            #del pretrained_dict['classifier[3].classification_layer.2.bias']
            new_model_dict.update(pretrained_dict)

            if args.multi_gpu :
                model.module.load_state_dict(new_model_dict)
            else :
                model.load_state_dict(new_model_dict)

            print('[Notice] Loading Pretrain Model ready.\n')

        else :
            # Set iterator
            iterator = Iterator(args, model, dataloaders, criterion=torch.nn.CrossEntropyLoss(), pretrain=True)
            print('[Notice] Iterator ready. - Pre-train Mode')

            # PRETrain the model
            for epoch in range(args.pretrain_epochs):
                print('\nPretrain Epoch: [{:03d}/{:03d}] '.format(epoch+1, args.pretrain_epochs)+'-'*30)
                epoch_start = time.time()
                #=======================================================================
                iterator.train()
                #=======================================================================
                epoch_end = time.time()
                print("[Notice] Epoch elapsed time: {0:.3f} sec".format(epoch_end - epoch_start))
                if iterator.checkpointer.stop_training: break

        # Set iterator
        optimizer = None
        if args.fine_tuning:
            params = get_classification_params(args, model)
            optimizer = get_optimizer(args, model, params)
        iterator = Iterator(args, model, dataloaders, optimizer=optimizer, pretrain=False)
        print('[Notice] Iterator ready. - Fine-tuning Mode')

    else :
        # Set iterator
        iterator = Iterator(args, model, dataloaders)
        print('[Notice] Iterator ready. - Normal Mode')

    # Train the model
    for epoch in range(args.epochs):
        print('\nEpoch: [{:03d}/{:03d}] '.format(epoch+1, args.epochs)+'-'*30)
        epoch_start = time.time()
        #=======================================================================
        iterator.train()
        #=======================================================================
        epoch_end = time.time()
        print("[Notice] Epoch elapsed time: {0:.3f} sec".format(epoch_end - epoch_start))
        if iterator.checkpointer.stop_training: break

    # Test the model
    if args.num_gpus > 0 and (args.local_rank % args.num_gpus == 0):
        if os.path.exists(args.BEST_CHECKPOINT_PATH):
            print()
            checkpoint_dict = torch.load(args.BEST_CHECKPOINT_PATH, map_location=args.device)
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            print('[Notice] Load best score model: {} epochs ({} iters)'.format(checkpoint_dict['current_epoch'],
                                                                                checkpoint_dict['current_iter']))
            iterator.test()

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
