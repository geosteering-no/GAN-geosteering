import numpy as np
import torch
import torch.nn.functional as F
# from functorch import jacrev # alternative jacobean computation

import matplotlib.pyplot as plt

from image_to_log import set_global_seed

import image_to_log
import vector_to_image

class FullModel:
    def __init__(self, latent_size, gan_save_file, proxi_input_shape, proxi_output_shape, proxi_save_file,
                 proxi_scalers=None, gan_output_height=64, num_img_channels=6, device='cpu', gan_correct_orientation=False):

        if proxi_scalers is None:
            # error
            # todo: fix this
            raise ValueError("proxi_scalers must be provided")


        self.gan_evaluator = vector_to_image.GanEvaluator(gan_save_file, latent_size, number_chanels=num_img_channels, device=device)
        self.gan_output_height = gan_output_height
        self.gan_channels = num_img_channels
        self.gan_correct_orientation = gan_correct_orientation
        # when we load the wrapper the gan model is already in eval mode

        # pad the image to the input size
        self.pad_top = (proxi_input_shape[1] - gan_output_height) // 2
        self.pad_bottom = (proxi_input_shape[1] - gan_output_height) - self.pad_top

        self.rh_mult  = torch.tensor([4, 171, 55, 0, 0, 0], dtype=torch.float32).to(device).view(1, -1, 1, 1)
        self.rv_mult = torch.tensor([4*5, 171*2, 55, 0, 0, 0], dtype=torch.float32).to(device).view(1, -1, 1, 1)

        # Initialize EM model
        self.proxi_input_shape = proxi_input_shape
        self.em_model = image_to_log.EMProxy(proxi_input_shape, proxi_output_shape,
                                             checkpoint_path=proxi_save_file, scaler=proxi_scalers).to(device)

        # in case the model needs to be put in the eval mode
        self.em_model.eval()
        self.em_model.requires_grad_(False)

    def convert_to_resistivity_format(self, images, index_vector):
        # add eval location
        one_hot = F.one_hot(index_vector, num_classes=images.shape[2]).float()  # [b, w, h]
        one_hot = one_hot.permute(0, 2, 1).unsqueeze(1)  # [b, 1, h, w]

        # reduce image to the index vector w
        images = images[:, :, :, 0:index_vector.shape[1]]  # [b, c, h, w]

        # convert index to one-hot vector
        rh = (images * self.rh_mult).sum(dim=1, keepdim=True)
        rv = (images * self.rv_mult).sum(dim=1, keepdim=True)

        resistivity = torch.cat([rh, rv, one_hot], dim=1)  # [b, 3, h, w]
        # now we need to scale it to the input dimensions

        b, c, h, w = resistivity.shape
        resistivity = resistivity.permute(0, 3, 1, 2)  # → [b, w, c, h]
        resistivity_flat = resistivity.reshape(b * w, c, h)  # → merge b and w
        resistivity_padded_flat = F.pad(resistivity_flat, (self.pad_top, self.pad_bottom), mode='replicate')  # pad h only

        # pad requires the input of the padding in two directions for the last dimensions in the reverse order
        # resistivity_padded = F.pad(resistivity, (self.pad_top, self.pad_bottom, 0, 0), mode='replicate')

        return resistivity_padded_flat

    def forward(self, x, index_vector, output_transien_results=False):
        """
        the output is flattened in batch and width dimensions
        """
        # Generate image from latent vector
        gan_output = self.gan_evaluator.eval(input_latent_ensemble=x, to_one_hot=True)
        if self.gan_correct_orientation:
            gan_output = gan_output.permute(0,1,3,2)

        resistivity_padded = self.convert_to_resistivity_format(gan_output, index_vector)

        # Convert image to log
        em_output = self.em_model.image_to_log(resistivity_padded)
        if not output_transien_results:
            return em_output
        else:
            return gan_output, resistivity_padded, em_output

if __name__ == "__main__":
    test_gradients = False
    # fix seeds
    seed = 777
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'  # for testing purposes, use CPU

    file_name = 'grdecl_15_15_1_60/netG_epoch_15000.pth'
    vec_size = 60
    # gan_evaluator = vector_to_image.GanEvaluator(file_name, vec_size, number_chanels=6, device=device)

    # Initialize model
    input_shape = (3, 128)
    output_shape = (6, 18)

    weights_folder = "../../UTA-proxy/training_results_85"
    scalers_folder = weights_folder
    # check path by converting string to path
    model_index = 770
    full_em_model_file_name = f'{weights_folder}/checkpoint_{model_index}.pth'

    full_model = FullModel(
        latent_size=vec_size,
        gan_save_file=file_name,
        proxi_save_file=full_em_model_file_name,
        proxi_scalers=scalers_folder,
        proxi_input_shape=input_shape,
        proxi_output_shape=output_shape,
        gan_output_height=64,
        num_img_channels=6,
        gan_correct_orientation=True,
        device=device
    )

    # example usage
    my_latent_vec_np = np.random.normal(size=vec_size)
    my_latent_tensor = torch.tensor(my_latent_vec_np, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and move to device
    # make index vector with all ints = 32
    index_tensor_bw = torch.full((1, 64), fill_value=32, dtype=torch.long).to(device)

    image, resistivity, logs = full_model.forward(my_latent_tensor, index_vector=index_tensor_bw, output_transien_results=True)
    pad_top = full_model.pad_top

    if test_gradients:
        my_latent_tensor = torch.tensor(my_latent_vec_np.tolist(), dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)
        index_tensor_bw = torch.full((1, 2), fill_value=32, dtype=torch.long).to(device)
        jacobean = torch.autograd.functional.jacobian(lambda x: full_model.forward(x, index_vector=index_tensor_bw),
                                                      my_latent_tensor,
                                                      create_graph=False,
                                                      vectorize=True)
        jacobean_np = jacobean.cpu().detach().numpy()

    logs_np = logs.cpu().detach().numpy()

    # the images are upside down, so we need to flip them
    image = image.flip([2])
    resistivity = resistivity.flip([2])

    # dump resistivity to numpy
    resistivity_np = resistivity.cpu().detach().numpy()
    np.savez('logs/resistivity.npz', resistivity=resistivity_np)

    cols, setups, log_types = logs_np.shape
    names = [
        'real(xx)', 'img(xx)',
        'real(yy)', 'img(yy)',
        'real(zz)', 'img(zz)',
        'real(xz)', 'img(xz)',
        'real(zx)', 'img(zx)',
        'USDA', 'USDP',
        'UADA', 'UADP',
        'UHRA', 'UHRP',
        'UHAA', 'UHAP'
    ]
    tool_configs = [f'{f} khz - {s} ft' for f, s in zip([6., 12., 24., 24., 48., 96.], [83., 83., 83., 43., 43., 43.])]

    for i in range(log_types):
        fig, (ax_img, ax_res, ax_logs) = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True, height_ratios=[1, 2, 6]
        )

        # plotting the image
        num_cols = image.shape[3]
        # note, that the image is rotated weirdly in the current training
        img = image[0, 0:3, :, :].permute(1, 2, 0).cpu().numpy()  # shape: [64, 64, 3]
        ax_img.imshow(img, extent=(-0.5, num_cols - 0.5, pad_top, pad_top+image.shape[2]),
                      aspect='auto', interpolation='none')
        ax_img.set_title("Facies image")
        ax_img.set_ylim(pad_top+image.shape[2], pad_top)

        # plotting resistivity
        img_res = resistivity[:, 0, :].T.cpu().numpy()  # shape: [H, W]
        ax_res.imshow(img_res, extent=(-0.5, num_cols - 0.5, 0, resistivity.shape[2]),
                      aspect='auto', interpolation='none', cmap='summer')
        ax_res.set_title("Resistivity input")


        # plotting logging locations
        mask = resistivity[:, 2, :].T.cpu().numpy()  # shape: [H, W]
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        rgba[..., 0:3] = 0.0  # black color
        rgba[..., 3] = (mask != 0).astype(np.float32)  # alpha: 1 for non-zero, 0 for zero

        ax_res.imshow(rgba, extent=(-0.5, num_cols - 0.5, 0, resistivity.shape[2]),
                      aspect='auto', interpolation='none')
        ax_res.set_ylim(resistivity.shape[2], 0)

        # plotting the logs
        ax_logs.set_title(names[i])
        logs_to_plot = logs_np[:, :, i]  # take the first batch and first channel
        for j, config in enumerate(tool_configs):
            ax_logs.plot(logs_to_plot[:, j], label=config)

        # saving
        fig.savefig(f'logs/log_{i}_{device}.png', bbox_inches='tight', dpi=300)

        # break

        if i == 0:
            ax_logs.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            fig.savefig(f'logs/log_{i}_{device}_with_legend.png', bbox_inches='tight', dpi=300)

    plt.show()



