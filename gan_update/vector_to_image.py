import matplotlib.pyplot as plt

import dcgan
import utils as myutils
import torch
import numpy as np
import os
import image_to_log
from PIL import Image, ImageOps

import random





class GanEvaluator:
    def __init__(self, load_file_name, vec_size, output_size=64, number_chanels=6, device='cpu', num_gpus=0):
        """

        """
        # TODO take from input
        noBN = False
        imageSize = output_size

        input_vector_size = vec_size

        num_gf_model_size = 64
        # ndf = 64
        number_chanels = number_chanels
        n_extra_layers = 0

        if noBN:
            netG = dcgan.DCGAN_G_nobn(imageSize, input_vector_size, number_chanels, num_gf_model_size, num_gpus, n_extra_layers).to(
                device)
        else:
            netG = dcgan.DCGAN_G(imageSize, input_vector_size, number_chanels, num_gf_model_size, num_gpus, n_extra_layers).to(
                device)
        netG.apply(myutils.weights_init)
        print('Loading GAN from {}'.format(load_file_name))
        netG.load_state_dict(torch.load(load_file_name, map_location='cpu'))
        netG.eval()

        self.netG = netG
        self.input_vector_size = input_vector_size
        self.number_channels = number_chanels
        self.image_size = imageSize
        self.device = device

    def eval(self, input_ensemble: np.ndarray = None, input_vec=None, to_one_hot=False, output_np=True):
        """
        Evaluate gan and produce an image as numpy array
        :param input_vec: 
        :return: 
        """
        # fixed_noize_vec = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1).to(device)
        if input_ensemble is not None:
            matrix_size = input_ensemble.shape
            input_tensor_2d = torch.from_numpy(input_ensemble).float().to(self.device)
            input_tensor = input_tensor_2d.transpose(0,1).unsqueeze(2).unsqueeze(3)
        elif input_vec is not None:
            vec_size = len(input_vec)
            input_tensor_1d = torch.from_numpy(input_vec).float().to(self.device)
            input_tensor = input_tensor_1d.reshape(1, vec_size, 1, 1)
        else:
            raise ValueError("Either input_ensemble or input_vec must be provided.")

        with torch.no_grad():
            x_fake = self.netG(input_tensor).detach()

        if to_one_hot:
            one_hot = torch.nn.functional.one_hot(torch.argmax(x_fake[:,:3,:,:], dim=1), num_classes=3).float()
            # permute to fit the original structure
            one_hot = one_hot.permute(0, 3, 1, 2)
            x_fake[:, :3, :, :] = one_hot

        if not output_np:
            # this will always be 4d tensor
            return x_fake

        x_fake_np = x_fake.cpu().numpy()

        if input_vec is not None:
            x_fake_np = x_fake_np.reshape(self.number_channels, self.image_size, self.image_size)

        return x_fake_np


def image_to_columns(tensor_4d: torch.Tensor, do_plot=False):
    # let's assume tensor of size [1, 6, 64, 64]
    # let's try splitting it in columns of size 32 (dim=2)

    columns = torch.chunk(tensor_4d.permute(0,1,3,2), chunks=tensor_4d.shape[3], dim=3)  # dim=3 is Width
    # plt.figure()
    # plt.imshow(tensor_4d[0, 0:3, :, :].permute(1,2,0).cpu().numpy(), aspect='auto', interpolation='none')
    # plt.show()


    if do_plot:
        plt.figure()
        gap = 2
        h = columns[0].shape[2]

        for i, column in enumerate(columns):
            img_col = column[0, 0:3, :, :].permute(1,2,0).cpu().numpy()
            plt.imshow(img_col, extent=(i * (1 + gap), i * (1 + gap) + 1, 0, h), aspect='auto', interpolation='none')
        plt.xlim(0, len(columns)*(1 + gap))
        plt.ylim(0, h)
        plt.show()

    return columns


def set_global_seed(seed: int):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    pass