import os
import numpy as np
import torch
from PIL import Image, ImageOps

from evaluation import evaluate_geological_realism
import vector_to_image

def trace_autograd_tree(output, max_depth=200):
    seen = set()
    def _recurse(fn, depth):
        if fn is None or fn in seen or depth > max_depth:
            return
        seen.add(fn)
        indent = "│   " * depth
        branch_note = f" ⎯⎯ branching ({len(fn.next_functions)} paths)" if len(fn.next_functions) > 1 else ""
        print(f"{indent}├── {type(fn).__name__}{branch_note}")
        for next_fn, _ in fn.next_functions:
            _recurse(next_fn, depth + 1)
    print("Autograd Graph:")
    _recurse(output.grad_fn, 0)

image_folder = 'images_test'

result_file = os.path.join(image_folder, '{}.csv'.format('generated'))
result_handler = open(result_file, 'a+')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vector_to_image.set_global_seed(42)

file_name = 'grdecl_15_15_1_60/netG_epoch_15000.pth'
vec_size = 60
gan_evaluator = vector_to_image.GanEvaluator(file_name, vec_size, number_chanels=6, device=device)

counter = 0

for _ in range(30):
    my_vec = np.random.normal(size=vec_size)
    my_tensor = torch.tensor(my_vec.tolist(), dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)  # Add batch dimension and move to device
    result = gan_evaluator.eval(input_latent_ensemble=my_tensor, to_one_hot=True, output_np=False)

    # this is for debugging purposes, uncomment to see the autograd tree
    # trace_autograd_tree(result)
    # uncomment to see the jacobian calculation
    # with torch.autograd.detect_anomaly():
    # Calculate jacobian
    # my_tensor_for_grad = torch.tensor(my_vec.tolist(), dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)
    # jacobian = torch.autograd.functional.jacobian(
    #     lambda x: gan_evaluator.eval(input_latent_ensemble=x, to_one_hot=True, output_np=False),
    #     my_tensor_for_grad)

    columns = vector_to_image.image_to_columns(result)

    image_data = result[0, 0:3, :, :].detach().permute(1 ,2 ,0).cpu().numpy()
    patch_normalized = np.rot90((np.clip(image_data, 0, 1) * 255).astype(np.uint8))

    # Convert to PIL image and save
    img = Image.fromarray(patch_normalized)
    img = ImageOps.flip(img)
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



