import os
import numpy as np
import torch
import random

import torch.nn.functional as F

from mymodel import EMConvModel


class EMProxy(EMConvModel):
    def __init__(self, input_shape, output_shape, checkpoint=None, device='cpu', scaler=None):
        super(EMProxy, self).__init__(input_shape, output_shape)
        checkpoint = torch.load(f'{weights_folder}/{checkpoint_name}', map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        self.scaler = scaler

    def image_and_index_to_log(self, column_input, tool_index):
        # convert tool index to one-hot vector (same size as column_input)
        one_hot_index = F.one_hot(tool_index, num_classes=column_input.shape[1])
        # now we need to stack it on top of the column_input
        one_hot = one_hot_index.unsqueeze(1)  # [B, c, L]
        stacked_input = torch.cat([column_input, one_hot], dim=1)  # [B, c+1, L]

        return self.eval(stacked_input)

    def image_to_log(self, input_tensor):
        return self.eval(input_tensor)

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

    # Initialize model
    input_shape = (3, 128)
    output_shape = (6, 18)

    model = EMProxy(input_shape, output_shape).to(device)

    weights_folder = "../../UTA-proxy/training_results"
    # check path by converting string to path
    weights_folder_path = os.path.abspath(weights_folder)
    print(f'Using weights from {weights_folder_path}')

    model_index = 770

    checkpoint_name = f"checkpoint_{model_index}.pth"
    try:
        checkpoint = torch.load(f'{weights_folder}/{checkpoint_name}', map_location=device)
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_name} not found in {weights_folder}.")
        exit()

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()  # Set to evaluation mode

    print("Model loaded successfully.")

    # Example usage
    column_input = torch.randn(1, 6, 64, 64)  # Example input tensor
    tool_index = torch.tensor([0])  # Example tool index

    log_output = model.image_and_index_to_log(column_input, tool_index)
    print(log_output)
