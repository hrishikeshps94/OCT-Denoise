import glob,os
from multiprocessing import set_forkserver_preload
import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T

class DenoiseDataset(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DenoiseDataset, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = []
        for dir_name,_,filenames in os.walk(data_dir):
            for filename in filenames:
                self.train_fns.append(os.path.join(dir_name,filename))   
        self.train_fns.sort()
        self.transforms = T.Compose([T.RandomCrop(size=(self.patch,self.patch)),T.PILToTensor(),T.ConvertImageDtype(dtype=torch.float32)])
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = self.transforms(im)
        return im

    def __len__(self):
        return len(self.train_fns)


class ValDenoiseDataset(Dataset):
    def __init__(self, data_dir):
        super(ValDenoiseDataset, self).__init__()
        self.data_dir = data_dir
        # self.gt_files = []
        self.input_files = []
        for dir_path,_,filenames in os.walk(data_dir):
            dir_name = dir_path.split('/')[-2]
            for filename in filenames:
                # if dir_name=='gt':
                #     self.gt_files.append(os.path.join(dir_path,filename))
                if dir_name == 'input':
                    self.input_files.append(os.path.join(dir_path,filename))
        # self.gt_files.sort()
        self.input_files.sort()
        self.transforms = T.ToTensor()
        self.PIL_Transform = T.PILToTensor()
        # self.crop = T.CenterCrop([512,512])
        print('fetch {} samples for testing'.format(len(self.input_files)))

    def __getitem__(self, index):
        # fetch image
        # gt = self.gt_files[index]
        input = self.input_files[index]
        gt = '/'.join(input.split('/')[0:-1])
        gt = gt.replace('input','gt')
        gt = gt+'.tif'
        gt_im = Image.open(gt)
        input_im = Image.open(input)
        # gt_im = self.crop(gt_im)
        # input_im = self.crop(input_im)
        gt_im_255 = self.PIL_Transform(gt_im)
        ###Add padding for generalisation ####################
        input_im_255 = self.PIL_Transform(input_im)
        input_im_train = self.transforms(input_im)
        return input_im_train,input_im_255,gt_im_255

    def __len__(self):
        return len(self.input_files)

# class ValDenoiseDataset(Dataset):
#     def __init__(self, data_dir):
#         super(ValDenoiseDataset, self).__init__()
#         self.data_dir = data_dir
#         self.gt_files = []
#         self.input_files = []
#         for dir_path,_,filenames in os.walk(data_dir):
#             dir_name = dir_path.split('/')[-1]
#             for filename in filenames:
#                 if dir_name=='GT':
#                     self.gt_files.append(os.path.join(dir_path,filename))
#                 elif dir_name == 'input':
#                     self.input_files.append(os.path.join(dir_path,filename))
#         self.gt_files.sort()
#         self.input_files.sort()
#         self.transforms = T.ToTensor()
#         self.PIL_Transform = T.PILToTensor()
#         self.crop = T.CenterCrop([512,512])
#         print('fetch {} samples for testing'.format(len(self.gt_files)))

#     def __getitem__(self, index):
#         # fetch image
#         gt = self.gt_files[index]
#         input = self.input_files[index]
#         gt_im = Image.open(gt)
#         input_im = Image.open(input)
#         gt_im = self.crop(gt_im)
#         input_im = self.crop(input_im)
#         gt_im_255 = self.PIL_Transform(gt_im)
#         ###Add padding for generalisation ####################
#         input_im_255 = self.PIL_Transform(input_im)
#         input_im_train = self.transforms(input_im)
#         return input_im_train,input_im_255,gt_im_255

#     def __len__(self):
#         return len(self.gt_files)