import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

home = os.path.expanduser("~") # os independent home

# local load of additional modules.
prefix = ''
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/KERNEL')
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append(prefix + '../gan-geosteering')

from vector_to_image import GanEvaluator


# def dynamic_programming_old(weights, start_point):
#     n, m = weights.shape
#     dp = np.full((n, m), -np.inf)  # Initialize with -inf for unvisited cells
#     path = np.zeros((n, m, 2), dtype=int)  # Store (prev_row, prev_col) for backtracking
#
#     # Initialize the start point
#     start_row, start_col = start_point
#     dp[start_row, start_col] = weights[start_row, start_col]
#
#     # Fill the DP table, considering horizontal and limited vertical moves from the start point
#     for j in range(start_col + 1, m):
#         for i in range(n):
#             # From directly left (i, j-1)
#             if j > start_col and dp[i, j - 1] + weights[i, j] > dp[i, j]:
#                 dp[i, j] = dp[i, j - 1] + weights[i, j]
#                 path[i, j] = (i, j - 1)
#
#             # From above (i-1, j) if not in the first row
#             if i > 0 and dp[i - 1, j] + weights[i, j] > dp[i, j]:
#                 dp[i, j] = dp[i - 1, j] + weights[i, j]
#                 path[i, j] = (i - 1, j)
#
#             # From below (i+1, j) if not in the last row
#             if i < n - 1 and dp[i + 1, j] + weights[i, j] > dp[i, j]:
#                 dp[i, j] = dp[i + 1, j] + weights[i, j]
#                 path[i, j] = (i + 1, j)
#
#     # Identify the optimal endpoint in the last column
#     end_row = np.argmax(dp[:, -1])
#     max_value = dp[end_row, -1]
#
#     # Backtrack to find the optimal path starting from the best endpoint
#     optimal_path = []
#     current = (end_row, m - 1)
#     while current[1] > start_col:  # Continue until reaching the starting column
#         optimal_path.append(current)
#         current = tuple(path[current])
#
#     optimal_path.append(start_point)  # Add the start of the path
#     optimal_path.reverse()  # Reverse the path to start from the optimal starting point
#
#     return dp, max_value, optimal_path



def dynamic_programming1(weights, start_point, discount_factor=1.0,
                         di_vector=None,
                         cost_vector=None):
    """
    :param weights:
    :param start_point:
    :param discount_factor:
    :param di_vector: defines possible vertical shifts in the trajectory
    :return:
    """
    # todo implement the discount factor
    # would the discout vector work if we go forward?

    if di_vector is None:
        di_vector = [0, -1, 1]
    if cost_vector is None:
        cost_vector = [1., 1., 1.]
    n, m = weights.shape
    path = np.ones((n, m, 2), dtype=int)*-1
    # path stores (next_row, next_col) or -1, -1 for backtracking

    # initialize the DP matrix with zeros
    # the cells meaning will be the max possible reward starting from that cell
    # we will not inclclude the cell's weight itself
    # since it is initialized with zeroes we never use negative values
    dp = np.full_like(weights, 0.0, dtype=float)  # Initialize with -inf for unvisited cells

    start_row, start_col = start_point

    # Fill the DP table from the last column back to the current position
    for j in range(m - 2, start_col - 1, -1):
        for i in range(n):
            for k, di in enumerate(di_vector):
                next_i = i+di
                drilling_direction_cost = cost_vector[k]
                # check if the next cell is in range
                if 0 <= next_i < n:
                    proposed_value = dp[next_i, j + 1] + weights[next_i, j + 1] - drilling_direction_cost
                    # todo check for discount here
                    # todo add drilling cost here
                    if proposed_value > dp[i, j]:
                        dp[i, j] = proposed_value
                        path[i, j] = (next_i, j+1)

    # # Initialize the start point as provided
    # dp[start_row, start_col] = weights[start_row, start_col]
    # # Fill the DP table, considering horizontal and diagonal moves from the start point
    # for j in range(start_col + 1, m):
    #     for i in range(n):
    #         # From directly left (i, j-1)
    #         if j > 0 and dp[i, j - 1] + weights[i, j] > dp[i, j]:
    #             dp[i, j] = dp[i, j - 1] + weights[i, j]
    #             path[i, j] = (i, j - 1)
    #
    #         # Diagonal upper-left (i-1, j-1)
    #         if i > 0 and j > 0 and dp[i - 1, j - 1] + weights[i, j] > dp[i, j]:
    #             dp[i, j] = dp[i - 1, j - 1] + weights[i, j]
    #             path[i, j] = (i - 1, j - 1)
    #
    #         # Diagonal lower-left (i+1, j-1)
    #         if i < n - 1 and j > 0 and dp[i + 1, j - 1] + weights[i, j] > dp[i, j]:
    #             dp[i, j] = dp[i + 1, j - 1] + weights[i, j]
    #             path[i, j] = (i + 1, j - 1)
    # Trace back the path from the best endpoint
    # end_row = np.argmax(dp[:, -1])
    # current = (end_row, m - 1)

    optimal_path = []
    optimal_path.append(start_point)
    current = tuple(path[start_row, start_col, :])

    while current[1] > 0:
        optimal_path.append(current)
        current = tuple(path[current])

    # while current[1] > start_col:  # Backtrack from the end column to the start column
    #     optimal_path.append(current)
    #     current = tuple(path[current])


    # optimal_path.reverse()  # Reverse the path to start from the optimal starting point

    # return dp, dp[end_row, -1], optimal_path
    return dp, dp[start_row, start_col], optimal_path


# gan_file_name = os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')
gan_file_name = '../gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth'

gan_vec_size = 60
gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)


def evaluate_earth_model(gan_evaluator, single_realization):
    earth_model = gan_evaluator.eval(input_vec=single_realization)
    rounded_model = np.where(earth_model >= 0, 1, 0)
    value_for_channel = {
    1: 1,   # Weight for channel body
    2: 0.5  # Weight for crevasse
    }

    result_matrix = calculate_body_sizes(rounded_model, value_for_channel)

    return result_matrix


def create_weighted_image(normalized_rgb):
    weights = np.array([-0.1, 1, 0.5])
    return np.dot(normalized_rgb, weights)


def perform_dynamic_programming(weighted_image, start_point, discount_factor=1.0,
                                di_vector=None,
                                cost_vector=None):
    if di_vector is None:
        di_vector = [0, -1, 1]
    # todo implement the discount factor
    dp = dynamic_programming1(weighted_image, start_point,
                              di_vector=di_vector,
                              cost_vector=cost_vector)  # Assume dynamic_programming1 is defined as before
    return dp


def plot_results(weighted_image, optimal_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    norm = Normalize(vmin=-0.1, vmax=1)
    ax.imshow(weighted_image, cmap='viridis', aspect='auto', norm=norm)
    path_rows, path_cols = zip(*optimal_path)
    ax.plot(path_cols, path_rows, 'k-', linewidth=2)
    ax.set_title('Result with Optimal Path', fontsize=18)
    plt.tight_layout()
    plt.show()


def process_prior_and_plot_results(single_realization, start_point, plot_path=False, discount_factor=1.0,
                                   di_vector=None,
                                   cost_vector=None):
    """

    :param single_realization:
    :param start_point:
    :param plot_path:
    :return:
    """
    if di_vector is None:
        di_vector = [0, -1, 1]
    # todo implement the discount factor
    normalized_rgb = evaluate_earth_model(gan_evaluator, single_realization)
    weighted_image = create_weighted_image(normalized_rgb)

    dp_matrix, max_path_value, optimal_path = perform_dynamic_programming(weighted_image, start_point,
                                                                          di_vector=di_vector,
                                                                          cost_vector=cost_vector)

    if plot_path:
        plot_results(weighted_image, optimal_path)


    # TODO note the weighted image in the output
    return dp_matrix, max_path_value, weighted_image, optimal_path


def process_matrix(single_realization, optimal_path, best_point, best_paths):
    fig, ax = plt.subplots(figsize=(10, 5))
    normalized_rgb = evaluate_earth_model(gan_evaluator, single_realization)
    weighted_image = create_weighted_image(normalized_rgb)

    if weighted_image is not None and optimal_path is not None:
        norm = Normalize(vmin=-0.1, vmax=1)
        ax.imshow(weighted_image, cmap='viridis', aspect='auto', norm=norm)
        path_rows, path_cols = zip(*optimal_path)
        ax.plot(path_cols, path_rows, 'k-', linewidth=2)  # 'k-' is for black line
        ax.set_title('Result with Optimal Path', fontsize=18)

    if best_paths is not None:
        for idx, path in enumerate(best_paths):
            # Assuming path is a list of row indices; generate column indices based on the length of the path
            if path[1] >= best_point[0]:
                path_cols = range(best_point[1], best_point[1] + len(path))  # Corrected range calculation
                ax.plot(path_cols, path, color='red')  # Label correctly with the index

        ax.set_title('Best Paths from DP Matrices', fontsize=18)
        ax.legend()

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('best.png')
    plt.close()  # Close the plot window to avoid displaying it in GUI

def calculate_body_sizes(single_earth_model_2d, value_for_channel=None):
    if value_for_channel is None:
        # Define the default weights for the channels
        value_for_channel = {
            1: 1,   # Weight for channel body
            2: 0.5  # Weight for crevasse
        }


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



# Example of calling the renamed function with the prior data
if __name__ == '__main__':
    prior_path = '/home/AD.NORCERESEARCH.NO/krfo/OneDrive/DISTINGUISH/ECMOR_study/RunFolder/debug_analysis_step_1.npz'
    prior = np.load(prior_path, allow_pickle=True)['state'][()]['m']
    start_point = (0, 0)
    process_prior_and_plot_results(prior[:,50], start_point, plot_path=True)