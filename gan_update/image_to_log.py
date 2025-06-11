import os
import numpy as np
import torch
import random

import torch.nn.functional as F
from udar_proxi.model_scaler import MinMaxScaler

from udar_proxi.mymodel import EMConvModel


class EMProxy(EMConvModel):
    def __init__(self, input_shape, output_shape, checkpoint_path=None, device='cpu', scaler=None, copy_to_dir=None):
        super(EMProxy, self).__init__(input_shape, output_shape)
        if 'https://' in checkpoint_path:
            # load from URL
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        if scaler is None:
            # error
            raise ValueError("Scaler must be provided")
        elif isinstance(scaler, MinMaxScaler):
            self.scaler = scaler.to(device)
        else:
            # presume that's a path to the scaler file
            self.scaler = MinMaxScaler(scaler_file_path=scaler, device=device)
        if copy_to_dir is not None:
            import shutil
            import os
            shutil.copy(checkpoint_path, os.path.join(copy_to_dir, os.path.basename(checkpoint_path)))
            self.scaler.save_scalers(copy_to_dir)
            print('The EM model has been saved to ' + copy_to_dir)

    def to(self, device):
        """
        Override to move the model and scaler to the specified device.
        """
        super(EMProxy, self).to(device)
        self.scaler.to(device)
        return self

    def image_and_index_to_log(self, column_input, tool_index):
        """
        column_input: should be of shape [Batch/columns, 2 channels, column_heilght]. That is [B, c, H]
        tool_index: should be of shape [Batch/columns]. That is [B]
        """
        # convert tool index to one-hot vector (same size as column_input)
        one_hot_index = F.one_hot(tool_index, num_classes=column_input.shape[2])
        # now we need to stack it on top of the column_input
        one_hot = one_hot_index.unsqueeze(1)  # [B, c, H]
        stacked_input = torch.cat([column_input, one_hot], dim=1)  # [B, c+1, H]

        # scaling
        scaled_input = self.scaler.scale_input(stacked_input)
        output = self.forward(scaled_input)
        scaled_output = self.scaler.unscale_output(output)

        return scaled_output

    def image_to_log(self, column_input_tensor):
        #scaling
        scaled_input = self.scaler.scale_input(column_input_tensor)
        output =  self.forward(scaled_input)
        scaled_output = self.scaler.unscale_output(output)

        return scaled_output

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed = 777
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_folder = "../../UTA-proxy/training_results_85"
    scaler_folder = weights_folder
    # check path by converting string to path
    weights_folder_path = os.path.abspath(weights_folder)
    print(f'Using weights from {weights_folder_path}')

    model_index = 770

    checkpoint_name = f"checkpoint_{model_index}.pth"
    checkpoint_path = f'{weights_folder}/{checkpoint_name}'
    # try:
    #     checkpoint = torch.load(f'{weights_folder}/{checkpoint_name}', map_location=device)
    # except FileNotFoundError:
    #     print(f"Checkpoint {checkpoint_name} not found in {weights_folder}.")
    #     exit()
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize model
    input_shape = (3, 128)
    output_shape = (6, 18)

    model = EMProxy(input_shape, output_shape, checkpoint_path=checkpoint_path,
                    scaler=scaler_folder,
                    copy_to_dir='../../vector_to_log_weights/em'
                    ).to(device)

    # model.eval()  # Set to evaluation mode unnecessary here, as it is already done in the constructor

    print("Model loaded successfully.")

    # Example usage
    image = torch.randn(1, 6, 128, 64).to(device)  # Example input tensor
    column_input = image.squeeze(0).permute(2,0,1)  # Convert to [Batch, Channels, Width]

    column_input = column_input[:, 0:2, :]  # keep only the first two channels
    tool_index = torch.ones(64, dtype=torch.long).to(device) * 24  # Example tool index, should be of shape [Batch/columns]

    log_output = model.image_and_index_to_log(column_input, tool_index)
    print(log_output)
