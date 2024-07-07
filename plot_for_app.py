import numpy as np
from pathOPTIM import pathfinder
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
from scipy.interpolate import make_interp_spline


# plt.ion()
# plt.show(block=False)

import os,sys

home = os.path.expanduser("~") # os independent home


# local load of additional modules.
prefix = ''
sys.path.append(prefix + '../deep-borehole-inverse-problem/KERNEL')
sys.path.append(prefix + '../deep-borehole-inverse-problem/USER_SERGEY')
sys.path.append(prefix + '../gan-geosteering-prestudy-internal')

from input_output import read_config
from GAN import GanLog
from resitivity import get_resistivity_default
from vector_to_image import GanEvaluator
from DP import perform_dynamic_programming, evaluate_earth_model, create_weighted_image, earth_model_from_vector

def convert_facies_to_resistivity(single_facies_model):
    my_shape = single_facies_model.shape
    result = np.zeros((my_shape[1], my_shape[2]))
    for i in range(my_shape[1]):
        for j in range(my_shape[2]):
            result[i, j] = get_resistivity_default(single_facies_model[:, i, j])
    return result

# import warnings
# # Ignore FutureWarning and UserWarning
# warnings.filterwarnings(action='ignore', category=FutureWarning)
# warnings.filterwarnings(action='ignore', category=UserWarning)

# gan_file_name = os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering/f2020_rms_5_10_50_60/netG_epoch_4662.pth')
gan_file_name = '../gan-geosteering-prestudy-internal/f2020_rms_5_10_50_60/netG_epoch_4662.pth'
gan_vec_size = 60
gan_evaluator = GanEvaluator(gan_file_name, gan_vec_size)

global_extent = [0, 640, -16.25, 15.75]

def earth(state):
    # we need to switch to TkAgg to show GUI, something switches it to somethiong else
    matplotlib.use('TkAgg')
    plot_path = 'figures/'
    #check if the path exists and create it if not
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    saved_legend = False

    synth_truth = np.load('../gan-geosteering-prestudy-internal/saves/chosen_realization_C1.npz')['arr_0']
    true_earth_model_facies = gan_evaluator.eval(input_vec=synth_truth)
    true_resistivity_image = convert_facies_to_resistivity(true_earth_model_facies)


    # todo should we switch back to probability of sand ??? (use custom weights)
    post_earth = np.array(
        [create_weighted_image(earth_model_from_vector(gan_evaluator, state[:, el]), weights=[0., 1., 1.]) for el in
         range(state.shape[1])])

    return post_earth,true_earth_model_facies

    # creating the figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # visualizing the posterior
    norm = Normalize(vmin=0.0, vmax=1)
    ax.imshow(post_earth.mean(axis=0), cmap='tab20b', aspect='auto', norm=norm)


    # Define the step-sizes
    step_x = 10
    step_y = 0.5

    # Calculate the tick positions and labels
    x_ticks_positions = np.arange(0, 64, 10)
    x_ticks_labels = (x_ticks_positions - 32) * step_x

    # Define the specific y-axis labels
    y_ticks_labels =     [-10, -5, 0, 5, 10]
    y_ticks_labels_str = ['x090', 'x095', 'x100', 'x105', 'x110']
    # Calculate the corresponding y tick positions
    y_ticks_positions = [label / step_y + 0 for label in y_ticks_labels]

    # Set x-ticks and labels
    ax.set_xticks(x_ticks_positions)
    ax.set_xticklabels(x_ticks_labels)
    # Set y-ticks and labels
    #ax.set_yticks(y_ticks_positions)
    #ax.set_yticklabels(y_ticks_labels_str)

    return fig,ax
