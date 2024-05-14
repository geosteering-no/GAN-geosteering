import numpy as np
import sys
import matplotlib.pyplot as plt

prefix = ''
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/KERNEL')
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append(prefix + '../gan-geosteering')

from vector_to_image import GanEvaluator

gan_file_name = '../gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth'
gan_vec_size = 60



gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)

prior = np.load('prior.npz')['m']
print(prior.shape)

single_realization = prior[:,0]
print(single_realization.shape)

earth_model = gan_evaluator.eval(input_vec=single_realization)
print(earth_model.shape)
print(earth_model[:,:,0])

rounded_model = np.where(earth_model >= 0, 1, 0)
print(rounded_model[:,:,0])

# # shale
# plt.figure(10)
# plt.imshow(rounded_model[0,:,:])
# plt.colorbar()
#
# # channel
# plt.figure(11)
# plt.imshow(rounded_model[1,:,:])
# plt.colorbar()
#
# # crevase
# plt.figure(12)
# plt.imshow(rounded_model[2,:,:])
# plt.colorbar()


tensor = rounded_model

# Initialize the result tensor with zeros


channel_for_channel_body = 1
channel_for_crevasse = 2

def calculate_body_sizes(single_earth_model_2d):
    # renaming
    tensor = single_earth_model_2d

    result_matrix = np.zeros_like(tensor[1, :, :], dtype=float)

    # Calculate connected channel-body sizes
    for w in range(tensor.shape[2]):
        channel_body_sizes = np.zeros(tensor.shape[1])
        count = 0
        for h in range(tensor.shape[1]):
            if tensor[channel_for_channel_body, h, w] != 0:  # Assuming non-zero means part of the channel body
                count += 1
            else:
                # Update the entire connected component with the thickness count
                if count > 0:
                    for k in range(h - count, h):
                        channel_body_sizes[k] = count
                count = 0
        # Ensure the last component is updated
        if count > 0:
            for k in range(h - count + 1, h + 1):
                channel_body_sizes[k] = count

        # Assign the calculated sizes to the result tensor
        # print(channel_body_sizes)
        result_matrix[:, w] = channel_body_sizes
        # print(result_matrix[:, w])
    return result_matrix


result_matrix = calculate_body_sizes(rounded_model)

almost_zero = 2 ** -2
visualization_matrix = (np.log2(result_matrix+almost_zero))

# Visualizing the original and result tensors for the first channel
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(tensor[channel_for_channel_body, :, :], cmap='viridis')
axes[0].set_title('Original Channel 1')
fig.colorbar(im0, ax=axes[0], orientation='vertical')


im1 = axes[1].imshow(visualization_matrix, cmap='viridis')
axes[1].set_title('Result Channel 1')
fig.colorbar(im1, ax=axes[1], orientation='vertical')

plt.show()
