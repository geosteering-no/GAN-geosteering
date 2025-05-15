from __future__ import print_function
from __future__ import division

import argparse
import random


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import csv
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable
import os

from skimage.transform import resize as img_resize
from scipy.io import loadmat
import numpy as np
from torch.utils.data.dataset import Dataset
import time

from grdecl_loader import read_grdecl


def _convert_to_zero_one(d):
    if d < 1.5:
        return 0.0
    else:
        return 1.0


def _porous_converter(id_float, porosity):
    id = int(id_float)
    out = []
    if id == 0:
        out = [0.0, 0.0]
    elif id == 2:
        out = [0.0, 1.0]
    else:
        # todo consider making the range more accurate
        out = [1.0, porosity*2]
    return out


def _porous_converter_3d(id_float, porocity):
    id = int(id_float)
    n = id
    out = []
    for i in range(2):
        out.append(n % 2 + 0.0)
        n //= 2
    out.append(porocity)
    return out


def _porous_converter_6d(id_float, porosity):
    id = int(id_float)
    out = np.zeros(6, dtype=float)
    out[id] = 1.0
    out[id+3] = porosity
    return out


def _porous_converter_6d_3d(id_float, porosity):
    hei, cha, wid = porosity.shape
    out = np.zeros((hei, wid, 6, cha), dtype=float)
    x, z, y = np.indices((hei, cha, wid))
    id_int = id_float.astype(int)
    x, y, z, id_int = x.flatten(), y.flatten(), z.flatten(), id_int.flatten()
    out[x, y, id_int, z] = 1.0
    out[x, y, id_int + 3, z] = porosity.flatten()
    out = out.reshape(hei, wid, 6, cha)

    return out

class FaciesConverter:
    def __init__(self, shift_index=1):
        self.shift_index = shift_index

    def convert_to_channels_2(self, d):
        n = int(d) - self.shift_index
        out = []
        for i in range(2):
            out.append(n % 2 + 0.0)
            n //= 2
        return out

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class BarPrinter:
    def __init__(self):
        self.prev = -100

    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        filledLength = int(length * iteration // total)
        if filledLength == self.prev:
            return
        else:
            self.prev = filledLength
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        bar = fill * filledLength + '-' * (length - filledLength)
        print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), flush=True)
        # Print New Line on Complete
        if iteration == total:
            print()


class CustomDatasetFromRMS(Dataset):
    def __init__(self, filename, height, width, transforms=None, channels=1, constant_axis=0, min_facies_id=1,
                 do_flip=True,
                 porous=0,
                 stride_x=1,
                 stride_y=1,
                 max_samples=100500600,
                 max_files=100500):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.channels = channels
        self.data = self._build_image_patches(filename, height, width, nc=channels,
                                              constant_axis=constant_axis,
                                              min_facies_number=min_facies_id,
                                              do_flip=do_flip,
                                              porous=porous,
                                              max_samples=max_samples,
                                              max_files=max_files,
                                              stride_x=stride_x,
                                              stride_y=stride_y)
        self.labels = np.zeros(self.data.shape[0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def _read_images_from_folder(self,
                                 foldername,
                                 converter=_convert_to_zero_one,
                                 converter_argument_count=1,
                                 constant_axis=0,
                                 do_flip=True,
                                 max_files=100500):
        list=[]
        for x in os.listdir(foldername):
            if (x.startswith(".")) :
                continue
            if (x.endswith(".png")):
                continue
            if (x.endswith(".txt")):
                continue
            if (x.endswith(".pptx")):
                continue

            list.append(x)
            if len(list)>=max_files:
                break

        bar_prtinter = BarPrinter()
        image_list = []
        total = len(list)
        cur = 0
        for filename in list:
            img = self._read_image_from_file(foldername+"/"+filename,
                                             converter=converter,
                                             converter_arguments=converter_argument_count,
                                             constant_axis=constant_axis, do_flip=do_flip)
            cur += 1
            bar_prtinter.printProgressBar(cur, total)
            if img is not None:
                image_list.append(img)
                #break
        return image_list

    def _read_image_from_file(self, filename,
                              converter=_convert_to_zero_one,
                              converter_arguments=1,
                              constant_axis=0, do_flip=True):
        list2d = []
        list_cur = []
        with open(filename, 'r') as file:
            prev_point = np.full(3, np.inf)
            for line in file:
                str_parts = list(filter(lambda x: len(x)>0, line.strip().split(' ')))
                if (len(str_parts) >= 4):
                    values = list(map(lambda x: float(x), str_parts))
                    np_point = np.array(values)[0:3]
                    changed = np.abs(np_point - prev_point) > 1e-2
                    prev_point = np_point
                    if np.sum(changed) > 1:
                        if len(list_cur) > 0:
                            list2d.append(list_cur)
                            #print(len(list_cur))
                            #print(list_cur)
                            if changed[constant_axis]:
                                return None
                        list_cur = []
                    if converter_arguments==1:
                        list_cur.append(converter(values[3]))
                    elif converter_arguments==2:
                        list_cur.append(converter(values[3], values[4]))
                    else:
                        raise Exception("not implemented")
                    #print("changed:  ", different)
                    #print(np_point)
                    t = 0
            #print("the end")
            #print(list2d)
            if len(list_cur) > 0:
                list2d.append(list_cur)
                print(len(list_cur))
                # print(list_cur)
        #img = np.transpose(np.array(list2d))
        img = np.array(list2d)
        #plt.imshow(img[:,:,1])
        #plt.show()

        img = np.transpose(img)
        #plt.imshow(img[1,:,:])
        #plt.show()

        if (do_flip):
            img = np.flip(img, 1)
        tmp0 = img < 0
        tmp1 = img > 1
        img_shape = img.shape
        img2 = np.empty((img_shape[1], img_shape[2], 3), dtype=float)
        img2[:, :, 0] = img[0, :, :]
        img2[:, :, 1] = img[1, :, :]
        img2[:, :, 2] = 0.0
        # fig, axs = plt.subplots()
        # plt.imshow(img2)
        # fig.suptitle(filename, fontsize=16)
        # plt.savefig(filename+".png")

        #plt.show()

        return img

    def _build_image_patches(self,
                             filename,
                             height,
                             width,
                             nc,
                             constant_axis=0,
                             min_facies_number=1,
                             do_flip=True,
                             porous=0,
                             stride_x=1,
                             stride_y=1,
                             max_samples=100500600,
                             max_files=100500):
        t0 = time.time()
        print('Start building the image patches')
        #mat_dict = loadmat('out_{}.mat'.format(int(image_idx)))
        #image_np = mat_dict['out']
        #image_np is a 2d array dtype uint8


        # if self.channels==2:
        #     image_np = self._read_image_from_file(filename, converter=_convert_to_channels_2, constant_axis=0)
        # else:
        #     image_np = self._read_image_from_file(filename)

        if porous == 1:
            images_np = self._read_images_from_folder(filename,
                                                      converter=_porous_converter,
                                                      converter_argument_count=2,
                                                      constant_axis=constant_axis,
                                                      do_flip=do_flip,
                                                      max_files=max_files)
            #finish
        elif porous == 2:
            images_np = self._read_images_from_folder(filename,
                                                      converter=_porous_converter_3d,
                                                      converter_argument_count=2,
                                                      constant_axis=constant_axis,
                                                      do_flip=do_flip,
                                                      max_files=max_files)
        elif porous == 6:
            images_np = self._read_images_from_folder(filename,
                                                      converter=_porous_converter_6d,
                                                      converter_argument_count=2,
                                                      constant_axis=constant_axis,
                                                      do_flip=do_flip,
                                                      max_files=max_files)
        elif self.channels==2:
            converter_class = FaciesConverter(min_facies_number)
            images_np = self._read_images_from_folder(filename,
                                                      converter=converter_class.convert_to_channels_2,
                                                      constant_axis=constant_axis,
                                                      do_flip=do_flip)
        else:
            images_np = self._read_images_from_folder(filename)

        num_images = len(images_np)
        image_shape = images_np[0].shape
        if len(image_shape)==2:
            image_size_x, image_size_y = image_shape
        else:
            image_ch, image_size_x, image_size_y = image_shape
        # aspect_ratio = image_size_y / image_size_x
        # patch_size_x = 128
        # patch_size_y = int(patch_size_x * aspect_ratio)
        upscale_factor = 1
        assert height == width
        patch_size_x = patch_size_y = height * upscale_factor
        # todo change stride as needed
        # stride_x = stride_y = 1
        n_images_x = (image_size_x - patch_size_x - 1) // stride_x + 1
        n_images_y = (image_size_y - patch_size_y - 1) // stride_y + 1

        # huge memory but loaded only parts are loaded on the GPU sequentially
        total_samples = min(max_samples, n_images_x * n_images_y * num_images)
        patches_np = np.empty(
            (total_samples,
             nc,
             int(patch_size_x / upscale_factor),
             int(patch_size_y / upscale_factor)
             ),
            dtype='float32')

        progress_bar = BarPrinter()
        local_idx = 0
        for img in images_np:
            # printProgressBar(local_idx, total_samples)
            if local_idx >= total_samples:
                break
            for idx_i in range(0, image_size_x - patch_size_x, stride_x):
                progress_bar.printProgressBar(local_idx, total_samples)
                if local_idx >= total_samples:
                    break
                for idx_j in range(0, image_size_y - patch_size_y, stride_y):
                    # progress_bar.printProgressBar(local_idx, total_samples)
                    if local_idx >= total_samples:
                        break
                    for idx_c in range(nc):
                        temp_img = img[idx_c, idx_i:idx_i + patch_size_x, idx_j:idx_j + patch_size_y]
                        temp_img = img_resize(
                            temp_img, (temp_img.shape[0] / upscale_factor, temp_img.shape[1] / upscale_factor),
                            preserve_range=True)  # mode='reflect',  , anti_aliasing=True)
                        patches_np[local_idx, idx_c, :, :] = temp_img
                    local_idx += 1

        # convertion to -1.0 1.0
        # Was
        #   patches_np[patches_np >= 0.5] = 1.0
        #   patches_np[patches_np < 0.5] = -1.0
        patches_np = (patches_np - 0.5) * 2.0
        patches_np[patches_np < -1.0] = -1.0
        patches_np[patches_np > 1.0] = 1.0
        # TODO check if there is in between
        xx = np.unique(patches_np)
        # need to shuffle the data
        num_patches = patches_np.shape[0]
        rand_indices = np.random.permutation(num_patches)
        patches_np = patches_np[rand_indices, :]

        # finished image preparation
        # return torch.tensor(patches_np)
        print('Finished building the image patches in {} sec of size {}'.format((time.time() - t0), patches_np.shape))

        return patches_np

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data[index, :])

        # # Transform image to tensor
        # if self.transforms is not None:
        #     img_as_tensor = self.transforms(img_as_np)
        # Return image and the label
        return (torch.tensor(img_as_np), single_image_label)

    def __len__(self):
        return self.data.shape[0]


class CustomDatasetFromGRD(Dataset):
    def __init__(self, filename, height, width, transforms=None, channels=1, constant_axis=0, min_facies_id=1,
                 do_flip=True,
                 porous=0,
                 stride_x=1,
                 stride_y=1,
                 max_samples=100500600,
                 max_files=100500):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.channels = channels
        self.data = self._build_image_patches(filename, height, width, nc=channels,
                                              constant_axis=constant_axis,
                                              min_facies_number=min_facies_id,
                                              do_flip=do_flip,
                                              max_samples=max_samples,
                                              max_files=max_files,
                                              stride_x=stride_x,
                                              stride_y=stride_y)
        self.labels = np.zeros(self.data.shape[0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def _read_images_from_folder(self,
                                 foldername,
                                 converter=_convert_to_zero_one,
                                 converter_argument_count=1,
                                 constant_axis=0,
                                 do_flip=False,
                                 max_files=100500):
        list=[]
        for x in os.listdir(foldername):
            if (x.endswith(".grdecl")):
                list.append(x)
            if len(list)>=max_files:
                break

        bar_prtinter = BarPrinter()
        image_list = []
        total = len(list)
        cur = 0
        for filename in list:
            img = self._read_image_from_file(foldername+"/"+filename,
                                             converter=converter,
                                             converter_arguments=converter_argument_count,
                                             constant_axis=constant_axis, do_flip=do_flip)
            cur += 1
            bar_prtinter.printProgressBar(cur, total)
            if img is not None:
                for k in range(img.shape[-1]):
                    image_list.append(img[:, :, :, k])
                #break
        return image_list

    def _read_image_from_file(self, filename,
                              converter=_convert_to_zero_one,
                              converter_arguments=1,
                              constant_axis=0, do_flip=True):
        porosity, facies = read_grdecl(filename)
        img = _porous_converter_6d_3d(facies, porosity)

        img = np.transpose(img, [2, 0, 1, 3])

        if (do_flip):
            img = np.flip(img, 1)

        # for i in range(100):
        #     plt.imshow(img[1,:,:,i])
        #     plt.savefig(r'C:\NORCE_Projects\DISTINGUISH\code\src\gan\tmp_images\{}-0.png'.format(i))
        #     img2 = np.empty((400, 400, 3), dtype=float)
        #     img2[:, :, 0] = img[0, :, :, i]
        #     img2[:, :, 1] = img[1, :, :, i]
        #     img2[:, :, 2] = 0
        #     plt.savefig(r'C:\NORCE_Projects\DISTINGUISH\code\src\gan\tmp_images\{}-1.png'.format(i))

        # img_shape = img.shape
        # img2 = np.empty((img_shape[1], img_shape[2], 3), dtype=float)
        # img2[:, :, 0] = img[0, :, :]
        # img2[:, :, 1] = img[1, :, :]
        # img2[:, :, 2] = 0.0
        # fig, axs = plt.subplots()
        # plt.imshow(img2)
        # fig.suptitle(filename, fontsize=16)
        # plt.savefig(filename+".png")

        #plt.show()

        return img

    def _build_image_patches(self,
                             filename,
                             height,
                             width,
                             nc,
                             constant_axis=0,
                             min_facies_number=1,
                             do_flip=True,
                             stride_x=32,
                             stride_y=32,
                             max_samples=100500600,
                             max_files=100500):
        t0 = time.time()
        print('Start building the image patches')
        #mat_dict = loadmat('out_{}.mat'.format(int(image_idx)))
        #image_np = mat_dict['out']
        #image_np is a 2d array dtype uint8


        # if self.channels==2:
        #     image_np = self._read_image_from_file(filename, converter=_convert_to_channels_2, constant_axis=0)
        # else:
        #     image_np = self._read_image_from_file(filename)

        images_np = self._read_images_from_folder(filename)

        num_images = len(images_np)
        image_shape = images_np[0].shape
        if len(image_shape)==2:
            image_size_x, image_size_y = image_shape
        else:
            image_ch, image_size_x, image_size_y = image_shape
        # aspect_ratio = image_size_y / image_size_x
        # patch_size_x = 128
        # patch_size_y = int(patch_size_x * aspect_ratio)
        upscale_factor = 1
        assert height == width
        patch_size_x = patch_size_y = height * upscale_factor
        # todo change stride as needed
        # stride_x = stride_y = 1
        n_images_x = (image_size_x - patch_size_x - 1) // stride_x + 1
        n_images_y = (image_size_y - patch_size_y - 1) // stride_y + 1

        # huge memory but loaded only parts are loaded on the GPU sequentially
        total_samples = min(max_samples, n_images_x * n_images_y * num_images)
        print(f'Potential samples: {n_images_x * n_images_y * num_images}, but loaded {max_samples}')
        patches_np = np.empty(
            (total_samples,
             nc,
             int(patch_size_x / upscale_factor),
             int(patch_size_y / upscale_factor)
             ),
            dtype='float32')

        progress_bar = BarPrinter()
        local_idx = 0
        for img in images_np:
            # printProgressBar(local_idx, total_samples)
            if local_idx >= total_samples:
                break
            for idx_i in range(0, image_size_x - patch_size_x, stride_x):
                progress_bar.printProgressBar(local_idx, total_samples)
                if local_idx >= total_samples:
                    break
                for idx_j in range(0, image_size_y - patch_size_y, stride_y):
                    # progress_bar.printProgressBar(local_idx, total_samples)
                    if local_idx >= total_samples:
                        break
                    for idx_c in range(nc):
                        temp_img = img[idx_c, idx_i:idx_i + patch_size_x, idx_j:idx_j + patch_size_y]
                        temp_img = img_resize(
                            temp_img, (temp_img.shape[0] / upscale_factor, temp_img.shape[1] / upscale_factor),
                            preserve_range=True)  # mode='reflect',  , anti_aliasing=True)
                        patches_np[local_idx, idx_c, :, :] = temp_img
                    local_idx += 1

        # convertion to -1.0 1.0
        # Was
        #   patches_np[patches_np >= 0.5] = 1.0
        #   patches_np[patches_np < 0.5] = -1.0
        patches_np = (patches_np - 0.5) * 2.0
        patches_np[patches_np < -1.0] = -1.0
        patches_np[patches_np > 1.0] = 1.0
        # TODO check if there is in between

        # need to shuffle the data
        num_patches = patches_np.shape[0]
        rand_indices = np.random.permutation(num_patches)
        patches_np = patches_np[rand_indices, :]

        # finished image preparation
        # return torch.tensor(patches_np)
        print('Finished building the image patches in {} sec of size {}'.format((time.time() - t0), patches_np.shape))

        return patches_np

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data[index, :])

        # # Transform image to tensor
        # if self.transforms is not None:
        #     img_as_tensor = self.transforms(img_as_np)
        # Return image and the label
        return (torch.tensor(img_as_np), single_image_label)

    def __len__(self):
        return self.data.shape[0]


class CustomDatasetFromMAT(Dataset):
    def __init__(self, image_idx, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = self._build_image_patches(image_idx, height, width)
        self.labels = np.zeros(self.data.shape[0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def _build_image_patches(self, image_idx, height, width):
        t0 = time.time()
        print('Start building the image patches')
        mat_dict = loadmat('out_{}.mat'.format(int(image_idx)))
        image_np = mat_dict['out']
        image_size_x, image_size_y = image_np.shape
        # aspect_ratio = image_size_y / image_size_x
        # patch_size_x = 128
        # patch_size_y = int(patch_size_x * aspect_ratio)
        upscale_factor = 4
        assert height == width
        patch_size_x = patch_size_y = height * upscale_factor
        stride_x = stride_y = 32
        n_images_x = (image_size_x - patch_size_x - 1) // stride_x + 1
        n_images_y = (image_size_y - patch_size_y - 1) // stride_y + 1

        # huge memory but loaded only parts are loaded on the GPU sequentially
        patches_np = np.empty(
            (n_images_x * n_images_y,
             1,
             int(patch_size_x / upscale_factor),
             int(patch_size_y / upscale_factor)
             ),
            dtype='float32')

        local_idx = 0
        for idx_i in range(0, image_size_x - patch_size_x, stride_x):
            for idx_j in range(0, image_size_y - patch_size_y, stride_y):
                temp_img = image_np[idx_i:idx_i + patch_size_x, idx_j:idx_j + patch_size_y]
                temp_img = img_resize(
                    temp_img, (temp_img.shape[0] / upscale_factor, temp_img.shape[1] / upscale_factor),
                    preserve_range=True)  # mode='reflect',  , anti_aliasing=True)
                patches_np[local_idx, 0, :, :] = temp_img
                local_idx += 1
        patches_np[patches_np >= 0.5] = 1.0
        patches_np[patches_np < 0.5] = -1.0

        # need to shuffle the data
        num_patches = patches_np.shape[0]
        rand_indices = np.random.permutation(num_patches)
        patches_np = patches_np[rand_indices, :]

        # finished image preparation
        # return torch.tensor(patches_np)
        print('Finished building the image patches in {} sec of size {}'.format((time.time() - t0), patches_np.shape))

        return patches_np

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data[index, :])

        # # Transform image to tensor
        # if self.transforms is not None:
        #     img_as_tensor = self.transforms(img_as_np)
        # Return image and the label
        return (torch.tensor(img_as_np), single_image_label)

    def __len__(self):
        return self.data.shape[0]

def _get_color_mapping22():
    #a = np.array([[0., -1.],[-1., 0.],[0.,0.]])
    #a = np.array([[0., -1.], [-1., 0.]], dtype=np.float32)
    a = np.array([[-1., 0.], [0., -1]], dtype=np.float32)
    return torch.from_numpy(a)


def save_image(x_real, filename, device):
    orig_shape = x_real.shape
    nc = orig_shape[1]
    # x_save_real = np.empty((orig_shape[0], 3, orig_shape[2],orig_shape[3]), dtype='f')\
    x_save_real = torch.zeros([orig_shape[0], 3, orig_shape[2], orig_shape[3]], dtype=torch.float, device=device)
    # x_save_real[:, 0:orig_shape[1], :, :] = x_real[:, 0:orig_shape[1], :, :]

    # image is from -1 to 1
    if nc == 2:
        color_conversion_mat = _get_color_mapping22()
    elif nc ==6:
        pass


    try:
        if nc == 2:
            output = torch.einsum("abcd,bf->afcd", x_real, color_conversion_mat)
        elif nc == 6:
            output = x_real
        else:
            output = x_real
    except:
        output = x_real
    # from -1 to 1

    # output = output.mul(0.25)
    # mul 0.25 gives -0.25 to 0.25
    output = output.mul(0.5)
    # output = output.add(0.66)
    # add gives 0.5 to 1.0
    output = output.add(0.5)
    # add maps to 0 1

    if nc == 2:
        x_save_real[:, 0:orig_shape[1], :, :] = output[:, 0:orig_shape[1], :, :]
    if nc == 6:
        x_save_real[:, 0:3, :, :] = output[:, 0:3, :, :]
    #vutils.save_image(x_save_real.mul(0.25).add(0.4), filename)
    vutils.save_image(x_save_real, filename)

    if nc == 6:
        #print(torch.argmax(x_save_real, dim=1))
        x_save_real[:, 0:3, :, :] = output[:, 3:6, :, :]
        x_save_real = x_save_real.mul(2.7)
        #gives -0.25 .. 0.25
        #x_save_real.mul(2.0)
        #x_save_real.add(0.5)
        # print(x_save_real)
        second_file = filename.replace(".png", "_por.png")
        vutils.save_image(x_save_real, second_file)
        #x_save_real[:, 0:3, :, :] = output[:, 0:3, :, :]


