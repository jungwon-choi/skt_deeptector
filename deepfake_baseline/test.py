# Copyright © 2020 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import numpy as np
import argparse
import time
import os

from tools.utils import *
from tools.iterator import Iterator
from tools.args import TestArgs

#===============================================================================
parser = argparse.ArgumentParser(description='Deepfake Detector test code')
args = TestArgs().initialize(parser)
#===============================================================================

def main(args):
    # Fix the seed for reproducibility
    set_reproducibility(args)

    # Build the model
    model = get_model(args)
    checkpoint_dict = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    print('[Notice] Load best score model: {} epochs ({} iters)'.format(checkpoint_dict['current_epoch'],
                                                                            checkpoint_dict['current_iter']))
    print('[Notice] Model ready.')

    # Set environments dataLoaders
    envs = {}
    dataloaders = {}
    phase = 'test'
    envs[phase] = get_envs(args, phase=phase)
    dataloaders[phase] = get_env_dataloaders(envs[phase], train=False,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    for env_name in dataloaders[phase].keys():
        dataset = dataloaders[phase][env_name].dataset
        num_real_framse = len(dataset.all_real_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_real_frame)
        num_fake_framse = len(dataset.all_fake_frame_list) if dataset.random_sample_num is None else len(dataset.sampled_fake_frame)
        print('{:<6} {:<15}: real {:5d}, fake {:5d}'.format(phase, env_name, num_real_framse, num_fake_framse))
    print('[Notice] Dataloader ready.')

    # Set iterator
    iterator = Iterator(args, model, dataloaders)
    print('[Notice] Iterator ready.')

    # Test the model
    test_start = time.time()
    #=======================================================================
    iterator.test()
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
    # # MEASURE PERFORMANCE
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
    print("[Start time]", time.strftime('%c', time.localtime(time.time())))
    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("\n[Notice] Total elapsed time: {0:.3f} sec".format(end - start))
    print("[Complete time]", time.strftime('%c', time.localtime(time.time())), end='\n'*2)
