# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
import argparse
import random
import time
import os

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

    # Set environments dataLoaders
    envs = {}
    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        phase_batch_size = int(args.batch_size/len(args.domain_list)) if args.split_envs and phase=='train' else args.batch_size
        print('Get {:<5} dataloader '.format(phase)+'-'*26)
        envs[phase] = get_envs(args, phase=phase)
        dataloaders[phase] = get_env_dataloaders(envs[phase], train=(phase=='train'),
                                                batch_size=phase_batch_size,
                                                num_workers=args.num_workers)
        for env_name in dataloaders[phase].keys():
            dataset = dataloaders[phase][env_name].dataset
            num_real_framse = len(dataset.all_real_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_real_frame)
            num_fake_framse = len(dataset.all_fake_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_fake_frame)
            print('{:<6} {:<15}: real {:5d}, fake {:5d}'.format(phase, env_name, num_real_framse, num_fake_framse))
    print('[Notice] Dataloader ready.\n')

    # Set iterator
    iterator = Iterator(args, model, dataloaders)
    print('[Notice] Iterator ready.')

    # Train the model
    for epoch in range(args.epochs):
        print('\nEpoch: [{:03d}/{:03d}]'.format(epoch+1, args.epochs), '-'*30)
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
    print("[Start time]", time.strftime('%c', time.localtime(time.time())))
    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("\n[Notice] Total elapsed time: {0:.3f} sec".format(end - start))
    print("[Complete time]", time.strftime('%c', time.localtime(time.time())), end='\n'*2)
