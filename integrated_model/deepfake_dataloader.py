# Copyright © 2020-2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from albumentations import JpegCompression
import pdb

#===============================================================================
class Deepfake_Dataset(Dataset):
    """
    Dataloader for cropped face images of faceforensics++, DeepfakeTIMIT, UADFV, Celeb-DFv2.
    """
    damain_label_dict = {
        'FaceForensics': 0,
        'UADFV': 1,
        'DeepfakeTIMIT': 2,
        'CelebDF': 3,
    }
    split_idx_dict = { # Total, train/val, val/test
        'FaceForensics': (1000, 720, 860),  # 720 ids / 140 ids / 140 ids
        'UADFV': (49, 35, 42),              # 35 ids / 7 ids / 7 ids
        'DeepfakeTIMIT': (32, 22, 27),      # 22 ids / 5 ids / 5 ids
        'CelebDF': (62, 43, 52),            # 40 ids - 14 15 18 X / 9 ids / 10 ids
    }
    #===========================================================================
    def __init__(self, root, dataset_name, split_type='train', img_size=299,
                transform=None, num_classes=2, quality=100, real_sample_x=1, unfair_sample=False,
                sample_num=32, sample_stride=1, auto_stride=True, keep_last_frame=False, random_sample_num=None,
                with_id=False, with_path=False, wtih_domain_label=False, domain_label=None,
                verbose=False, debug=False, num_debug_sample=5,
                comp_type='raw', hq=True, infinite_mode=False, attack_test=False, phase_concat=False, phase_only=False, attack=False):
        ##########################################################################################################################
        self.root = root
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.img_size = img_size

        self.transform = transform
        self.num_classes = num_classes
        self.quality = quality

        self.sample_num = sample_num
        self.real_sample_x = real_sample_x
        self.sample_stride = sample_stride
        self.auto_stride = auto_stride
        self.keep_last_frame = keep_last_frame
        self.random_sample_num = random_sample_num
        self.unfair_sample = unfair_sample

        self.with_id = with_id
        self.with_path = with_path
        self.wtih_domain_label = wtih_domain_label
        self.domain_label = domain_label

        self.verbose = verbose
        self.debug = debug
        self.num_debug_sample = num_debug_sample

        # Dataset specific options
        self.comp_type = comp_type
        self.hq = hq
        self.attack_test = attack_test
        self.phase_concat = phase_concat
        self.phase_only = phase_only
        self.attack = attack

        # Dataloader options
        self.infinite_mode = infinite_mode
        self.total_iters = 10000000
        self.pool = {}
        for i in range(3):
            self.pool[i] = []
        if self.split_type != 'train':
            self.infinite_mode = False

        if self.domain_label is None:
            self.wtih_domain_label = False
        ########################################################################
        if self.quality < 100:
            self.compressor = JpegCompression(quality_lower=self.quality, quality_upper=self.quality, p=1)
        ########################################################################
        self.all_real_frame_list = self.get_frame_list(fake=0)
        self.all_fake_frame_list = self.get_frame_list(fake=1)
        if self.random_sample_num is None or self.random_sample_num == 0 or self.debug or (self.infinite_mode and self.split_type=='train'):
            self.video_frame_list = self.all_real_frame_list + self.all_fake_frame_list
            if self.infinite_mode:
                if self.unfair_sample:
                    self.reset_pool(fake=2)
                else:
                    self.reset_pool(fake=0)
                    self.reset_pool(fake=1)
        else:
            self.sample_frame_list()

    #===========================================================================
    def sample_frame_list(self):
        # print(self.dataset_name, len(self.all_real_frame_list), len(self.all_fake_frame_list ), flush=True)
        self.sampled_real_frame = random.sample(self.all_real_frame_list, round(self.random_sample_num/2))
        self.sampled_fake_frame = random.sample(self.all_fake_frame_list, round(self.random_sample_num/2))
        self.video_frame_list = self.sampled_real_frame + self.sampled_fake_frame

    #===========================================================================
    def reset_pool(self, fake):
        if fake == 2:
            self.pool[2] = list(range(len(self.video_frame_list)))
        elif fake == 1:
            self.pool[1] = list(range(len(self.all_fake_frame_list)))
        else:
            self.pool[0] = list(range(len(self.all_real_frame_list)))

    def get_sample_index(self, class_idx):
        if len(self.pool[class_idx]) == 0:
            self.reset_pool(fake=class_idx)
        idx = random.sample(self.pool[class_idx], 1)[0]
        self.pool[class_idx].remove(idx)
        return idx
    #===========================================================================
    def __len__(self):
        if self.infinite_mode:
            return self.total_iters
        else:
            return len(self.video_frame_list)

    #===========================================================================
    def __getitem__(self, index):
        if self.infinite_mode and self.split_type == 'train':
            if not self.unfair_sample:
                class_idx = np.random.choice([0, 1], 1, True, [0.5, 0.5])[0]
                idx = self.get_sample_index(class_idx)
                if class_idx:
                    video_frame_info = self.all_fake_frame_list[idx]
                else:
                    video_frame_info = self.all_real_frame_list[idx]
            else:
                idx = self.get_sample_index(2)
                video_frame_info = self.video_frame_list[idx]
        else:
            if self.attack_test:
                selected_list = random.choice([self.sampled_real_frame,self.sampled_fake_frame])
                video_frame_info = selected_list[index]
            else:
                video_frame_info = self.video_frame_list[index]

        frame = self.get_frame_from_image(path=video_frame_info['path'])

        if video_frame_info['label'] == 'REAL':
            label = torch.zeros(1)
        elif video_frame_info['label'] == 'FAKE':
            label = torch.ones(1)
        else:
            label = None

        if (label is not None) and self.num_classes == 2:
            label = label.long()

        if self.with_path and not self.wtih_domain_label:
            path = video_frame_info['path']
            return frame, label, path

        if self.wtih_domain_label:
            # domain_label = self.damain_label_dict[self.dataset_name]
            domain_label = self.domain_label
            if self.with_path:
                path = video_frame_info['path']
                return frame, label, domain_label, path
            return frame, label, domain_label

        else:
            return frame, label

    #===========================================================================
    def get_frame_from_image(self, path):
        image = Image.open(path)

        if self.quality < 100:
            frame = Image.fromarray(self.compressor(image=np.asarray(image))['image'])
        if self.transform is not None:
            frame = self.transform(image)
        if self.phase_concat or self.phase_only:
            image = image.convert('L')
            image = image.resize((self.img_size,self.img_size))
            np_img = np.array(image)
            # np_img = np.transpose(image, (2, 0, 1))
            frame_spectrum = np.fft.fft2(np_img)
            phase = np.angle(frame_spectrum)
            frame_phase = np.fft.ifft2(phase).real
            frame_phase = torch.from_numpy(frame_phase).float().unsqueeze(dim=0)
            if self.phase_only:
                return frame_phase
            if type(frame) is list:
                frame = [torch.cat([x, frame_phase], dim=0) for x in frame]
            else:
                frame = torch.cat([frame, frame_phase], dim=0)
        return frame

    #===========================================================================
    def get_frame_list(self, fake=True):
        frame_label = 'FAKE' if fake else 'REAL'

        # Set sub video directory
        if self.dataset_name == 'FaceForensics':
            subdir = (os.path.join('manipulated_sequences', 'Deepfakes',
                            self.comp_type, 'videos') if fake
                 else os.path.join('original_sequences', 'youtube',
                            self.comp_type, 'videos'))
        elif self.dataset_name == 'UADFV':
            subdir = 'fake' if fake else 'real'
        elif self.dataset_name == 'DeepfakeTIMIT':
            subdir = (os.path.join('fake', 'higher_quality'
                            if self.hq else 'lower_quality') if fake
                      else 'real')
        elif self.dataset_name == 'CelebDF':
            subdir = 'Celeb-synthesis' if fake else 'Celeb-real' # 'YouTube-real'

        # Get file list
        file_dir = os.path.join(self.root, subdir)
        file_list = os.listdir(file_dir)
        file_list.sort()

        # Get split index
        split_idx = self.split_idx_dict[self.dataset_name]

        # Select video files
        video_list = []
        for i in range(split_idx[0]):
            if self.split_type == 'train':
                if i >= split_idx[1]: break
            elif self.split_type == 'val':
                if i < split_idx[1] or i >= split_idx[2]: continue
            elif self.split_type == 'test':
                if i < split_idx[2]: continue

            if self.dataset_name == 'FaceForensics':
                video_list.append(file_list[i])
            elif self.dataset_name == 'UADFV':
                video_list.extend([file for file in file_list
                                    if '%04d'%i in file])
            elif self.dataset_name == 'DeepfakeTIMIT':
                video_list.extend([os.path.join(file_list[i], subvid)
                                    for subvid in os.listdir(os.path.join(file_dir, file_list[i]))])
            elif self.dataset_name == 'CelebDF':
                video_list.extend([file for file in file_list
                                    if ('id%d_i'%i if fake else 'id%d_'%i) in file])

        # Sample frames from each video
        video_frame_list = list()
        total_iter = len(video_list)

        video_frame_stride = self.sample_stride
        sample_num = self.sample_num
        if not fake and self.real_sample_x != 1:
            sample_num *= self.real_sample_x
            sample_num = int(sample_num)

        for ii, (video_file) in enumerate(video_list):
            if self.verbose:
                print("\r{} videos {:04}/{:04}".format(frame_label.lower(), ii+1,total_iter), end='', flush=True)

            video_file_path = os.path.join(file_dir, video_file)

            files = os.listdir(video_file_path)
            files.sort()
            frame_files = [file for file in files if file.endswith(".png")]
            video_total_frame = len(frame_files)

            if sample_num is not None:
                if sample_num > video_total_frame:
                    # print(video_file, video_total_frame, flush=True)
                    continue

                assert sample_num <= video_total_frame, \
                    'The number of frames of {} is less than {}'.format(video_file_path, sample_num)

                if self.auto_stride:
                    video_frame_stride = max(int(video_total_frame/sample_num),1)
            else:
                video_frame_stride = 1

            count = 0
            for frame_idx in range(0, video_total_frame, video_frame_stride):
                video_frame_list.append({'path':os.path.join(video_file_path, frame_files[frame_idx]),
                                         'label':frame_label,})
                count+=1
                if sample_num is not None:
                    if count == sample_num: break
                else:
                    if count == video_total_frame: break
                if frame_idx+1+video_frame_stride>video_total_frame:
                    if frame_idx == video_total_frame-1:
                        break
                    if self.keep_last_frame:
                        # To use up to the last frame
                        frame_idx = video_total_frame-1
                        video_frame_list.append({'path':os.path.join(video_file_path, frame_files[frame_idx]),
                                                 'label':frame_label,})
                        count+=1
                    break

            if self.debug:
                if ii > self.num_debug_sample-1:
                    break
        if self.verbose:
            print()
        return video_frame_list
