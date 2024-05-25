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

    saved_legend = False

    # todo load from the config file! @KriFos1
    # Define the origin
    origin_x = 0
    origin_y = 32
    drilled_path = [np.array([origin_y, origin_x])]
    num_decission_points = 63
    # for i in range(num_decission_points):
    for i in range(1):
        # Load the decision points
        checkpoint_at_step = np.load(f'estimate_decission_{i}.npz')
        state_vectors = checkpoint_at_step['m']
        position_at_step = checkpoint_at_step['pos']

        # todo remove test position
        position_at_step = np.array([33, 4])
        # position_at_step = np.array([6, 15])

        # todo remove test trajectory
        drilled_path.append(np.array([33, 2]))
        drilled_path.append(position_at_step)

        # this is the posterior
        # todo should we switch back to probability of sand ???
        post_earth = np.array(
            [create_weighted_image(evaluate_earth_model(gan_evaluator, state_vectors[:, el])) for el in
             range(ne)])  # range(state.shape[1])])

        next_optimal, _garbage_path = pathfinder().run(state_vectors, position_at_step)

        # creating the figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # visualizing the posterior
        norm = Normalize(vmin=0.0, vmax=1)
        im = ax.imshow(post_earth.mean(axis=0), cmap='tab20b', aspect='auto', norm=norm)
        cbar = plt.colorbar(im, ax=ax)

        # visualizing the drilled path
        path_rows, path_cols = zip(*(drilled_path))
        # todo consider creating smooth path. Does not work easily when few points.
        # # Creating a smooth curve
        # x = np.array(path_cols)
        # y = np.array(path_rows)
        # x_smooth = np.linspace(x.min(), x.max(), 300)
        # spl = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
        # y_smooth = spl(x_smooth)
        ax.plot(path_cols, path_rows,
                'k-', linewidth=3., label='Drilled path')
        ax.plot(path_cols, path_rows,
                'k*', label='Measurement locations')

        if next_optimal[0] is not None:

            # visualizing the next decision
            path_rows = [position_at_step[0], next_optimal[0]]
            path_cols = [position_at_step[1], next_optimal[1]]
            ax.plot(path_cols, path_rows,
                    'k:', label='Proposed direction')

            optimal_paths = [perform_dynamic_programming(post_earth[j, :, :], next_optimal,
                                                         cost_vector=pathfinder().get_cost_vector())[2] for j in range(ne)]

            # visualizing the optimal paths' remainders
            earth_height = post_earth.shape[2]
            max_height = earth_height - 0.5 - 0.01
            min_height = 0.0 - 0.5 + 0.01
            for j in range(ne):
                path_rows, path_cols = zip(*(optimal_paths[j]))
                row_list = [el + 0.2*np.random.randn() for el in path_rows]
                row_list_truncated = [el if el < max_height else max_height for el in row_list]
                row_list_truncated = [el if el > min_height else min_height for el in row_list_truncated]
                if j == 0:
                    ax.plot(path_cols, tuple(row_list_truncated),
                            'k--', linewidth=0.25, label='Further trajectory options')
                else:
                    ax.plot(path_cols, tuple(row_list_truncated),
                            'k--', linewidth=0.25)
            ax.set_title('Result with Optimal Path', fontsize=18)
            plt.tight_layout()


        # Define the step-sizes
        step_x = 10
        step_y = 0.5

        # Calculate the tick positions and labels
        x_ticks_positions = np.arange(0, 64, 10)
        x_ticks_labels = (x_ticks_positions - origin_x) * step_x

        # Define the specific y-axis labels
        y_ticks_labels =     [-10, -5, 0, 5, 10]
        y_ticks_labels_str = ['x090', 'x095', 'x100', 'x105', 'x110']
        # Calculate the corresponding y tick positions
        y_ticks_positions = [label / step_y + origin_y for label in y_ticks_labels]

        # Set the ticks and labels on the plot
        plt.xticks(x_ticks_positions, x_ticks_labels)
        plt.yticks(y_ticks_positions, y_ticks_labels_str)

        plt.savefig(f'{plot_path}mean_earth_{i}.png', bbox_inches='tight')
        plt.savefig(f'{plot_path}mean_earth_{i}.pdf', bbox_inches='tight')

        if not saved_legend:
            # Adding the legend outside the plot
            ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left')
            plt.savefig(f'{plot_path}legend.png', bbox_inches='tight')
            plt.savefig(f'{plot_path}legend.pdf', bbox_inches='tight')
            saved_legend = True


        drilled_path.append(checkpoint_at_step['pos'])# note that we compute another one during viosualization

        plt.show()

        # plot the mean GAN output for the current decision points

main()
