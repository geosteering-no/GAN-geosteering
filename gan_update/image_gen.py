import os
import numpy as np
from PIL import Image, ImageOps


from evaluation import evaluate_geological_realism
import vector_to_image

image_folder = 'images_test'

result_file = os.path.join(image_folder, '{}.csv'.format('generated'))
result_handler = open(result_file, 'a+')

device = 'cpu'
vector_to_image.set_global_seed(42)

file_name = 'grdecl_15_15_1_60/netG_epoch_15000.pth'
vec_size = 60
gan_evaluator = vector_to_image.GanEvaluator(file_name, vec_size, number_chanels=6)

counter = 0

for _ in range(3000):
    my_vec = np.random.normal(size=vec_size)
    result = gan_evaluator.eval(input_vec=my_vec, to_one_hot=True, output_np=False)

    columns = vector_to_image.image_to_columns(result)

    image_data = result[0, 0:3, :, :].permute(1 ,2 ,0).cpu().numpy()
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



