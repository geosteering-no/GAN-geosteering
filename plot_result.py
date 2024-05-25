import numpy as np
from pathOPTIM import pathfinder
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib


# plt.ion()
# plt.show(block=False)

import os,sys

home = os.path.expanduser("~") # os independent home


# local load of additional modules.
prefix = ''
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/KERNEL')
sys.path.append(prefix + '../deepEMdeepML2/deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append(prefix + '../gan-geosteering')

from input_output import read_config
from GAN import GanLog
from vector_to_image import GanEvaluator
from DP import perform_dynamic_programming, evaluate_earth_model, create_weighted_image


# import warnings
# # Ignore FutureWarning and UserWarning
# warnings.filterwarnings(action='ignore', category=FutureWarning)
# warnings.filterwarnings(action='ignore', category=UserWarning)

# gan_file_name = os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')
gan_file_name = '../gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth'
gan_vec_size = 60
gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)

global_extent = [0, 640, -16.25, 15.75]


ne = 250

def main():
    # we need to switch to TkAgg to show GUI, something switches it to somethiong else
    matplotlib.use('TkAgg')
    plot_path = 'figures/'
    #check if the path exists and create it if not
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # todo load from the config file! @KriFos1
    drilled_path = [np.array([32,0])]
    num_decission_points = 63
    # for i in range(num_decission_points):
    for i in range(1):
        # Load the decision points
        checkpoint_at_step = np.load(f'estimate_decission_{i}.npz')
        state_vectors = checkpoint_at_step['m']
        position_at_step = checkpoint_at_step['pos']

        # # todo remove test position
        position_at_step = np.array([33, 4])
        drilled_path.append(position_at_step)

        # this is the posterior
        post_earth = np.array(
            [create_weighted_image(evaluate_earth_model(gan_evaluator, state_vectors[:, el])) for el in
             range(ne)])  # range(state.shape[1])])

        next_optimal, _garbage_path = pathfinder().run(state_vectors, position_at_step)

        optimal_paths = [perform_dynamic_programming(post_earth[j,:,:], next_optimal)[2] for j in range(ne)]

        # creating the figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # visualizing the posterior
        norm = Normalize(vmin=0.0, vmax=1)
        im = ax.imshow(post_earth.mean(axis=0), cmap='tab20b', aspect='auto', norm=norm)
        cbar = plt.colorbar(im, ax=ax)

        # visualizing the optimal paths' remainders
        earth_height = post_earth.shape[2]
        max_height = earth_height - 0.5 - 0.01
        min_height = 0.0 - 0.5 + 0.01
        for j in range(ne):
            path_rows, path_cols = zip(*(drilled_path+optimal_paths[j]))
            row_list = [el + 0.2*np.random.randn() if c > len(drilled_path) else el for (c, el) in enumerate(path_rows)]
            row_list_truncated = [el if el < max_height else max_height for el in row_list]
            row_list_truncated = [el if el > min_height else min_height for el in row_list_truncated]
            ax.plot(path_cols, tuple(row_list_truncated),
                    'k-', linewidth=0.2)
        ax.set_title('Result with Optimal Path', fontsize=18)
        plt.tight_layout()
        # plt.savefig(f'{plot_path}mean_earth_{i}.png')

        drilled_path.append(checkpoint_at_step['pos'])# note that we compute another one during viosualization

        plt.show()

        # plot the mean GAN output for the current decision points

main()
