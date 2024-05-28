import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import os

# local load of additional modules.
prefix = ''
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/KERNEL')
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append(prefix + '../gan-geosteering')

from vector_to_image import GanEvaluator
from DP import perform_dynamic_programming, evaluate_earth_model

gan_file_name = os.path.join(prefix + '../gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')
gan_vec_size = 60



gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)

prior = np.load(prefix + '../gan-geosteering/saves/chosen_realization_C1.npz', allow_pickle=True)['arr_0']

single_realization = prior

earth_model = gan_evaluator.eval(input_vec=single_realization)
#
#
# def calculate_body_sizes(index_model_2d, value_for_channel=None):
#     """
#
#     :param index_model_2d: provides the index of the facies in each cell
#     :param value_for_channel:
#     :return:
#     """
#     if value_for_channel is None:
#         # Define the default weights for the channels
#         value_for_channel = {
#             1: 1,   # Weight for channel body
#             2: 0.5  # Weight for crevasse
#         }
#
#     # Initialize the result matrix with zeros
#     result_matrix = np.zeros_like(index_model_2d, dtype=float)
#
#     max_w = index_model_2d.shape[1]
#     max_h = index_model_2d.shape[0]
#
#     # Calculate connected channel-body sizes
#     for w in range(max_w):
#         channel_body_sizes = np.zeros(max_h)
#         count = 0
#         total_sum = 0
#         for h in range(max_h):
#             component_sum = 0
#             if index_model_2d[h,w] in value_for_channel:
#                 key = index_model_2d[h,w]
#                 component_sum += value_for_channel[key]
#             # for key in value_for_channel:
#             #     # todo change
#             #     if single_earth_model_2d[key, h, w] > 0:  # Check for the specified cell type
#             #         component_sum += value_for_channel[key]
#
#             if component_sum > 1:
#                 print('Warning, more than one likely component')
#
#             if component_sum > 0:
#                 total_sum += component_sum
#                 count += 1
#             else:
#                 # Update the entire connected component with the combined count using slicing
#                 if count > 0:
#                     channel_body_sizes[h - count:h] = total_sum
#                 count = 0
#                 total_sum = 0
#
#         # Ensure the last component is updated
#         if count > 0:
#             channel_body_sizes[h - count + 1:h + 1] = total_sum
#
#         # Assign the calculated sizes to the result tensor
#         result_matrix[:, w] = channel_body_sizes
#
#     return result_matrix
#
#
# # Define the weights for the channels
# value_for_channel = {
#     1: 1,   # Weight for channel body
#     2: 0.5  # Weight for crevasse
# }
# index_model = np.argmax(earth_model[0:3,:,:], axis=0)
# # Calculate the result matrix


result_matrix = evaluate_earth_model(gan_evaluator, single_realization)
    # calculate_body_sizes(index_model , value_for_channel))


#weighted_image = create_weighted_image(result_matrix)
print(result_matrix.shape)

# Set visualization_matrix to result_matrix
visualization_matrix = result_matrix



# Define the step-sizes
step_x = 10
step_y = 0.5

origin_x = 0
origin_y = 32

# Calculate the tick positions and labels
x_ticks_positions = np.arange(0, 64, 10)
x_ticks_labels = (x_ticks_positions - origin_x) * step_x

# Define the specific y-axis labels
y_ticks_labels = [-10, -5, 0, 5, 10]
y_ticks_labels_str = ['x090', 'x095', 'x100', 'x105', 'x110']

# Calculate the corresponding y tick positions
y_ticks_positions = [label / step_y + origin_y for label in y_ticks_labels]

# Visualizing the result tensor
plt.figure(figsize=(12, 5))

font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)

im = plt.imshow(visualization_matrix, cmap='plasma', aspect='auto')

# Set x-ticks and labels
plt.xticks(x_ticks_positions, x_ticks_labels)

# Set y-ticks and labels
plt.yticks(y_ticks_positions, y_ticks_labels_str)

# Adding title and colorbar
# plt.title('Result Channel Body with Thickness')
plt.colorbar(im, orientation='vertical')

# Save the figure
plt.savefig('figures/true_value.png', bbox_inches='tight', dpi=600)
plt.savefig('figures/true_value.pdf', bbox_inches='tight')

# Show the plot
plt.show()