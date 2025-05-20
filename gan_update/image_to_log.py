import os
import numpy as np
import torch
import random

from mymodel import EMConvModel


def image_to_log(
        column_input: torch.Tensor,
        tool_index: torch.Tensor):
    pass

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

    model = EMConvModel(input_shape, output_shape).to(device)

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

    exit()

    # Example usage
    column_input = torch.randn(1, 6, 64, 64)  # Example input tensor
    tool_index = torch.tensor([0])  # Example tool index

    log_output = image_to_log(column_input, tool_index)
    print(log_output)
