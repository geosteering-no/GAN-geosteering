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

    path = [np.array([32,0])]
    num_decission_points = 63
    for i in range(num_decission_points):
        # Load the decision points
        decission_point = np.load(f'estimate_decission_{i}.npz')
        post_earth = np.array(
            [create_weighted_image(evaluate_earth_model(gan_evaluator,decission_point['m'][:, el])) for el in
             range(ne)])  # range(state.shape[1])])

        optimal_path = [perform_dynamic_programming(post_earth[j,:,:], decission_point['pos'])[2] for j in range(ne)]

        fig, ax = plt.subplots(figsize=(10, 5))
        norm = Normalize(vmin=-0.1, vmax=1)
        ax.imshow(post_earth.mean(axis=0), cmap='tab20b', aspect='auto', norm=norm)
        for j in range(ne):
            path_rows, path_cols = zip(*(path+optimal_path[j]))
            ax.plot(path_cols, tuple([el + 0.2*np.random.randn() if c > len(path) else el for c,el in enumerate(path_rows)]),
                    'k-', linewidth=0.5)
        ax.set_title('Result with Optimal Path', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{plot_path}mean_earth_{i}.png')

        path.append(decission_point['pos'])


        # plot the mean GAN output for the current decision points

main()
