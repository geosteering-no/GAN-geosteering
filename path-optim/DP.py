import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

home = os.path.expanduser("~") # os independent home

# local load of additional modules.
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/KERNEL'))
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/USER_SERGEY'))
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering'))

from vector_to_image import GanEvaluator


def dynamic_programming(weights, start_point):
    n, m = weights.shape
    dp = np.full((n, m), -np.inf)  # Initialize with -inf for unvisited cells
    path = np.zeros((n, m, 2), dtype=int)  # Store (prev_row, prev_col) for backtracking

    # Initialize the start point
    start_row, start_col = start_point
    dp[start_row, start_col] = weights[start_row, start_col]

    # Fill the DP table, considering horizontal and limited vertical moves from the start point
    for j in range(start_col + 1, m):
        for i in range(n):
            # From directly left (i, j-1)
            if j > start_col and dp[i, j - 1] + weights[i, j] > dp[i, j]:
                dp[i, j] = dp[i, j - 1] + weights[i, j]
                path[i, j] = (i, j - 1)

            # From above (i-1, j) if not in the first row
            if i > 0 and dp[i - 1, j] + weights[i, j] > dp[i, j]:
                dp[i, j] = dp[i - 1, j] + weights[i, j]
                path[i, j] = (i - 1, j)

            # From below (i+1, j) if not in the last row
            if i < n - 1 and dp[i + 1, j] + weights[i, j] > dp[i, j]:
                dp[i, j] = dp[i + 1, j] + weights[i, j]
                path[i, j] = (i + 1, j)

    # Identify the optimal endpoint in the last column
    end_row = np.argmax(dp[:, -1])
    max_value = dp[end_row, -1]

    # Backtrack to find the optimal path starting from the best endpoint
    optimal_path = []
    current = (end_row, m - 1)
    while current[1] > start_col:  # Continue until reaching the starting column
        optimal_path.append(current)
        current = tuple(path[current])

    optimal_path.append(start_point)  # Add the start of the path
    optimal_path.reverse()  # Reverse the path to start from the optimal starting point

    return dp, max_value, optimal_path


def dynamic_programming1(weights, start_point):
    n, m = weights.shape
    dp = np.full_like(weights, -np.inf, dtype=float)  # Initialize with -inf for unvisited cells
    path = np.zeros((n, m, 2), dtype=int)  # Store (prev_row, prev_col) for backtracking

    # Initialize the start point as provided
    start_row, start_col = start_point
    dp[start_row, start_col] = weights[start_row, start_col]

    # Fill the DP table, considering horizontal and diagonal moves from the start point
    for j in range(start_col + 1, m):
        for i in range(n):
            # From directly left (i, j-1)
            if j > 0 and dp[i, j - 1] + weights[i, j] > dp[i, j]:
                dp[i, j] = dp[i, j - 1] + weights[i, j]
                path[i, j] = (i, j - 1)

            # Diagonal upper-left (i-1, j-1)
            if i > 0 and j > 0 and dp[i - 1, j - 1] + weights[i, j] > dp[i, j]:
                dp[i, j] = dp[i - 1, j - 1] + weights[i, j]
                path[i, j] = (i - 1, j - 1)

            # Diagonal lower-left (i+1, j-1)
            if i < n - 1 and j > 0 and dp[i + 1, j - 1] + weights[i, j] > dp[i, j]:
                dp[i, j] = dp[i + 1, j - 1] + weights[i, j]
                path[i, j] = (i + 1, j - 1)

    # Trace back the path from the best endpoint
    end_row = np.argmax(dp[:, -1])
    current = (end_row, m - 1)
    optimal_path = []

    while current[1] > start_col:  # Backtrack from the end column to the start column
        optimal_path.append(current)
        current = tuple(path[current])

    optimal_path.append(start_point)  # Include the start point
    optimal_path.reverse()  # Reverse the path to start from the optimal starting point

    return dp, dp[end_row, -1], optimal_path


gan_file_name = os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')
gan_vec_size = 60
gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)


def evaluate_earth_model(gan_evaluator, single_realization):
    earth_model = gan_evaluator.eval(input_vec=single_realization)
    earth_model = np.transpose(earth_model[0:3, :, :], axes=(1, 2, 0))
    scaled_image_data = (earth_model + 1) / 2
    norm_factors = np.sum(scaled_image_data, axis=-1, keepdims=True)
    norm_factors[norm_factors == 0] = 1
    normalized_rgb = scaled_image_data / norm_factors
    return normalized_rgb


def create_weighted_image(normalized_rgb):
    weights = np.array([-0.1, 1, 0.5])
    return np.dot(normalized_rgb, weights)


def perform_dynamic_programming(weighted_image, start_point):
    dp = dynamic_programming1(weighted_image, start_point)  # Assume dynamic_programming1 is defined as before
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


def process_prior_and_plot_results(single_realization, start_point, plot_path=False):
    normalized_rgb = evaluate_earth_model(gan_evaluator, single_realization)
    weighted_image = create_weighted_image(normalized_rgb)

    dp_matrix, max_path_value, optimal_path = perform_dynamic_programming(weighted_image, start_point)

    if plot_path:
        plot_results(weighted_image, optimal_path)

    return dp_matrix, max_path_value, optimal_path


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


# Example of calling the renamed function with the prior data
if __name__ == '__main__':
    prior_path = '/home/AD.NORCERESEARCH.NO/krfo/OneDrive/DISTINGUISH/ECMOR_study/RunFolder/debug_analysis_step_1.npz'
    prior = np.load(prior_path, allow_pickle=True)['state'][()]['m']
    start_point = (0, 0)
    process_prior_and_plot_results(prior[:,50], start_point, plot_path=True)