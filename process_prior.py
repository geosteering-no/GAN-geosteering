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
# print(earth_model[:,:,0])

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

def calculate_body_sizes(single_earth_model_2d, value_for_channel):
    # Initialize the result matrix with zeros
    result_matrix = np.zeros_like(single_earth_model_2d[1, :, :], dtype=float)

    # Calculate connected channel-body sizes
    for w in range(single_earth_model_2d.shape[2]):
        channel_body_sizes = np.zeros(single_earth_model_2d.shape[1])
        count = 0
        total_sum = 0
        for h in range(single_earth_model_2d.shape[1]):
            component_sum = 0
            for key in value_for_channel:
                if single_earth_model_2d[key, h, w] > 0:  # Check for the specified cell type
                    component_sum += value_for_channel[key]

            if component_sum > 1:
                print('Warning, more than one likely component')

            if component_sum > 0:
                total_sum += component_sum
                count += 1
            else:
                # Update the entire connected component with the combined count using slicing
                if count > 0:
                    channel_body_sizes[h - count:h] = total_sum
                count = 0
                total_sum = 0

        # Ensure the last component is updated
        if count > 0:
            channel_body_sizes[h - count + 1:h + 1] = total_sum

        # Assign the calculated sizes to the result tensor
        result_matrix[:, w] = channel_body_sizes

    return result_matrix

# Define the weights for the channels
value_for_channel = {
    1: 1,   # Weight for channel body
    2: 0.5  # Weight for crevasse
}

result_matrix = calculate_body_sizes(rounded_model, value_for_channel)

almost_zero = 2 ** -2
visualization_matrix = result_matrix

# Visualizing the original and result tensors for the specified channels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original channel body visualization
im0 = axes[0].imshow(rounded_model[1, :, :], cmap='jet', vmin=0, vmax=3)
axes[0].set_title('Original Channel Body')
fig.colorbar(im0, ax=axes[0], orientation='vertical')

# Original crevasse visualization
im1 = axes[1].imshow(rounded_model[2, :, :], cmap='jet', vmin=0, vmax=3)
axes[1].set_title('Original Crevasse')
fig.colorbar(im1, ax=axes[1], orientation='vertical')

# Result tensor visualization
im2 = axes[2].imshow(visualization_matrix, cmap='jet', vmin=0, vmax=3)
axes[2].set_title('Result Channel Body with Thickness')
fig.colorbar(im2, ax=axes[2], orientation='vertical')

plt.show()
