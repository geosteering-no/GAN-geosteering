import dcgan
import utils as myutils
import torch
import numpy as np
import os
import mcwd_converter
from PIL import Image


from evaluation import evaluate_geological_realism

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

    def eval(self, input_ensemble: np.ndarray = None, input_vec=None):
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

        with torch.no_grad():
            x_fake = self.netG(input_tensor).detach().numpy()

        if input_vec is not None:
            x_fake = x_fake.reshape(self.number_channels, self.image_size, self.image_size)

        return x_fake


# class GempyEvaluator:
#     # todo make sensible evaluation for GEMPY
#     def __init__(self, load_file_name, vec_size, output_size=64, number_chanels=2, device='cpu', num_gpus=0):
#         """
#
#         """
#         # TODO take from input
#         noBN = False
#         imageSize = output_size
#
#         input_vector_size = vec_size
#
#         num_gf_model_size = 64
#         # ndf = 64
#         number_chanels = number_chanels
#         n_extra_layers = 0
#
#         if noBN:
#             netG = dcgan.DCGAN_G_nobn(imageSize, input_vector_size, number_chanels, num_gf_model_size, num_gpus, n_extra_layers).to(
#                 device)
#         else:
#             netG = dcgan.DCGAN_G(imageSize, input_vector_size, number_chanels, num_gf_model_size, num_gpus, n_extra_layers).to(
#                 device)
#         netG.apply(myutils.weights_init)
#         netG.load_state_dict(torch.load(load_file_name, map_location='cpu'))
#
#         self.netG = netG
#         self.input_vector_size = input_vector_size
#         self.number_channels = 2
#         self.image_size = imageSize
#         self.device = device
#
#     # todo make sensible evaluation for GEMPY
#     def eval(self, input_ensemble: np.ndarray = None, input_vec=None):
#         """
#         Evaluate gan and produce an image as numpy array
#         :param input_vec:
#         :return:
#         """
#         # fixed_noize_vec = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1).to(device)
#         if input_ensemble is not None:
#             matrix_size = input_ensemble.shape
#             input_tensor_2d = torch.from_numpy(input_ensemble).float().to(self.device)
#             # TODO I am prety sure the shape is assumed to be wrong
#             input_tensor = input_tensor_2d.transpose(0,1).unsqueeze(2).unsqueeze(3)
#         elif input_vec is not None:
#             vec_size = len(input_vec)
#             input_tensor_1d = torch.from_numpy(input_vec).float().to(self.device)
#             input_tensor = input_tensor_1d.reshape(1, vec_size, 1, 1)
#
#         x_fake = self.netG(input_tensor).detach().numpy()
#         if input_vec is not None:
#             x_fake = x_fake.reshape(self.number_channels, self.image_size, self.image_size)
#
#         return x_fake


if __name__ == "__main__":
    image_folder = r'C:\NORCE_Projects\DISTINGUISH\Temp'

    result_file = os.path.join(image_folder, '{}.csv'.format('generated'))
    result_handler = open(result_file, 'a+')

    device = 'cpu'
    np.random.seed(42)

    file_name = r'C:\NORCE_Projects\DISTINGUISH\code\GAN-geosteering\gan_update\grdecl_32_32_50_60\netG_epoch_3996.pth'
    vec_size = 60
    gan_evaluator = GanEvaluator(file_name, vec_size, number_chanels=6)

    counter = 0

    for _ in range(3000):
        my_vec = np.random.normal(size=vec_size)
        result = gan_evaluator.eval(input_vec=my_vec)
        image_data = np.transpose(result[0:3, :, :], axes=(1, 2, 0))
        patch_normalized = np.rot90((np.clip(image_data, 0, 1) * 255).astype(np.uint8))

        # Convert to PIL image and save
        img = Image.fromarray(patch_normalized)
        filename = os.path.join(image_folder, '{}.png'.format(counter))
        img.save(filename)
        print(f"Image saved to {filename}")

        realism_score = evaluate_geological_realism(patch_normalized)
        result_handler.write('{}\n'.format(realism_score))
        result_handler.flush()
        print(realism_score)

        counter += 1

    result_handler.close()
    # plt.imshow(image_data,  aspect='auto')
    # plt.show()
    # column_of_pixels = result[:, :, 0]
    # mcwd_input = mcwd_converter.convert_to_mcwd_input(column_of_pixels, 32)
    # print(mcwd_input)



