import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from image_to_log import set_global_seed

import image_to_log
import vector_to_image

class FullModel:
    def __init__(self, latent_size, gan_save_file,
                 proxi_input_shape, proxi_output_shape, proxi_save_file,
                 gan_output_height=64, num_img_channels=6, device='cpu'):

        self.gan_evaluator = vector_to_image.GanEvaluator(gan_save_file, latent_size, number_chanels=num_img_channels, device=device)
        self.gan_output_height = gan_output_height
        self.gan_channels = num_img_channels
        # when we load the wrapper the gan model is already in eval mode

        # pad the image to the input size
        self.pad_top = (proxi_input_shape[1] - gan_output_height) // 2
        self.pad_bottom = (proxi_input_shape[1] - gan_output_height) - self.pad_top

        self.rh_mult  = torch.tensor([4, 171, 55, 0, 0, 0], dtype=torch.float32).to(device).view(1, -1, 1, 1)
        self.rv_mult = torch.tensor([4*5, 171*2, 55, 0, 0, 0], dtype=torch.float32).to(device).view(1, -1, 1, 1)

        # Initialize EM model
        self.proxi_input_shape = proxi_input_shape
        self.em_model = (image_to_log.EMProxy(proxi_input_shape, proxi_output_shape,
                                             checkpoint_path=proxi_save_file)
                         .to(device))
        # in case the model needs to be put in the eval mode
        self.em_model.eval()

    def convert_to_resistivity_format(self, image, index_vector):
        # convert index to one-hot vector
        rh = (image * self.rh_mult).sum(dim=1, keepdim=True)
        rv = (image * self.rv_mult).sum(dim=1, keepdim=True)
        # add eval location
        one_hot = F.one_hot(index_vector, num_classes=image.shape[2]).float()  # [b, w, h]
        one_hot = one_hot.permute(0, 2, 1).unsqueeze(1)  # [b, 1, h, w]

        resistivity = torch.cat([rh, rv, one_hot], dim=1)  # [b, 3, h, w]
        # now we need to scale it to the input dimensions

        b, c, h, w = resistivity.shape
        resistivity = resistivity.permute(0, 3, 1, 2)  # → [b, w, c, h]
        resistivity_flat = resistivity.reshape(b * w, c, h)  # → merge b and w
        resistivity_padded_flat = F.pad(resistivity_flat, (self.pad_top, self.pad_bottom), mode='replicate')  # pad h only

        # pad requires the input of the padding in two directions for the last dimensions in the reverse order
        # resistivity_padded = F.pad(resistivity, (self.pad_top, self.pad_bottom, 0, 0), mode='replicate')

        return resistivity_padded_flat

    def forward(self, x, index_vector):
        """
        the output is flattened in batch and width dimensions
        """
        # Generate image from latent vector
        gan_output = self.gan_evaluator.eval(input_latent_ensemble=x, to_one_hot=True)

        resistivity_padded = self.convert_to_resistivity_format(gan_output, index_vector)

        # Convert image to log
        em_output = self.em_model.image_to_log(resistivity_padded)

        return em_output

if __name__ == "__main__":
    # fix seeds
    seed = 777
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = 'grdecl_15_15_1_60/netG_epoch_15000.pth'
    vec_size = 60
    # gan_evaluator = vector_to_image.GanEvaluator(file_name, vec_size, number_chanels=6, device=device)

    # Initialize model
    input_shape = (3, 128)
    output_shape = (6, 18)

    weights_folder = "../../UTA-proxy/training_results"
    # check path by converting string to path
    model_index = 770
    full_em_model_file_name = f'{weights_folder}/checkpoint_{model_index}.pth'

    full_model = FullModel(
        latent_size=vec_size,
        gan_save_file=file_name,
        proxi_save_file=full_em_model_file_name,
        proxi_input_shape=input_shape,
        proxi_output_shape=output_shape,
        gan_output_height=64,
        num_img_channels=6,
        device=device
    )

    # example usage
    my_latent_vec_np = np.random.normal(size=vec_size)
    my_latent_tensor = torch.Tensor(my_latent_vec_np).unsqueeze(0).to(device)  # Add batch dimension and move to device
    # make index vector with all ints = 32
    index_tensor_bw = torch.full((1, 64), fill_value=32, dtype=torch.long).to(device)

    logs = full_model.forward(my_latent_tensor, index_vector=index_tensor_bw)

    logs_np = logs.cpu().detach().numpy()

    cols, setups, log_types = logs_np.shape
    names = [f'log_{i}' for i in range(log_types)]
    # todo figure out log names
    names[10] = 'my very special log'
    for i in range(log_types):
        plt.figure()
        plt.title(names[i])
        logs_to_plot = logs_np[:, :, i]  # take the first batch and first channel
        plt.plot(logs_to_plot)
    plt.show()



