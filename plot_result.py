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
from resitivity import get_resistivity_default
from vector_to_image import GanEvaluator
from DP import perform_dynamic_programming, evaluate_earth_model, create_weighted_image

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

    synth_truth = np.load('../gan-geosteering/saves/chosen_realization_C1.npz')['arr_0']
    true_earth_model_facies = gan_evaluator.eval(input_vec=synth_truth)
    true_resistivity_image = convert_facies_to_resistivity(true_earth_model_facies)

    # plot the resistivity:
    # Create the plot
    fig_res, ax_res = plt.subplots(figsize=(12, 5))
    # Use the 'viridis' colormap
    res_im = ax_res.imshow(true_resistivity_image, aspect='auto', cmap='summer', vmin=1, vmax=200)
    cbar = plt.colorbar(res_im, ax=ax_res)
    em_model_for_overlay_plotting_step = 18
    # plt.show()

    # todo load from the config file! @KriFos1
    # Define the origin
    labelled_index = 0
    origin_x = 0
    origin_y = 32
    # drilled_path = [np.array([origin_y, origin_x])]
    drilled_path = []
    num_decission_points = 63
    for i in range(0, num_decission_points):
        # for i in range(1):
        # Load the decision points
        checkpoint_at_step = np.load(f'estimate_decission_{i}.npz')
        state_vectors = checkpoint_at_step['m']
        start_position_at_step = checkpoint_at_step['pos']

        drilled_path.append(start_position_at_step)

        # # todo remove test position
        # position_at_step = np.array([33, 4])
        # # position_at_step = np.array([6, 15])
        #
        # # todo remove test trajectory
        # drilled_path.append(np.array([33, 2]))
        # drilled_path.append(position_at_step)

        # this is the posterior

        post_earth = np.array(
            [create_weighted_image(evaluate_earth_model(gan_evaluator, state_vectors[:, el])) for el in
             range(ne)])  # range(state.shape[1])])

        # todo should we switch back to probability of sand ??? (use custom weights)
        post_earth = np.array(
            [create_weighted_image(evaluate_earth_model(gan_evaluator, state_vectors[:, el]), weights=[0., 1., 1.]) for el in
             range(ne)])  # range(state.shape[1])])

        next_optimal, _garbage_path = pathfinder().run(state_vectors, start_position_at_step)

        # creating the figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # visualizing the posterior
        norm = Normalize(vmin=0.0, vmax=1)
        im = ax.imshow(post_earth.mean(axis=0), cmap='tab20b', aspect='auto', norm=norm)


        # visualizing the outline of the truth
        x = np.array(range(64))
        y = np.array(range(64))
        X, Y = np.meshgrid(x, y)
        Z = true_earth_model_facies[0, :, :]
        contour_style = 'dashed'
        contour_color = 'white'
        contour = plt.contour(X, Y, Z, levels=0, colors=contour_color,
                                linestyles=contour_style)
        # contour.collections[0].set_label('Outline of sand+crevasse in the true model')

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

        if i == em_model_for_overlay_plotting_step:
            ax_res.plot(path_cols, path_rows,
                    'k-', linewidth=3., label='Drilled path')
            ax_res.plot(path_cols, path_rows,
                    'k*', label='Measurement locations')
            # saving the true resistivity image with overlay
            # Define the step-sizes
            step_x = 10
            step_y = 0.5

            # Calculate the tick positions and labels
            x_ticks_positions = np.arange(0, 64, 10)
            x_ticks_labels = (x_ticks_positions - origin_x) * step_x

            # Define the specific y-axis labels
            y_ticks_labels = [-10, -5, 0, 5, 10]
            y_ticks_labels_str = ['x090', 'x095', 'x100', 'x105', 'x110']
            # Calculate the corresponding y tick positions
            y_ticks_positions = [label / step_y + origin_y for label in y_ticks_labels]

            # Set x-ticks and labels
            ax_res.set_xticks(x_ticks_positions)
            ax_res.set_xticklabels(x_ticks_labels)

            # Set y-ticks and labels
            ax_res.set_yticks(y_ticks_positions)
            ax_res.set_yticklabels(y_ticks_labels_str)

            fig_res.savefig(f'{plot_path}true_resistivity_{i}.png', bbox_inches='tight', dpi=600)
            fig_res.savefig(f'{plot_path}true_resistivity_{i}.pdf', bbox_inches='tight')

        if next_optimal[0] is not None:

            # visualizing the next decision
            path_rows = [start_position_at_step[0], next_optimal[0]]
            path_cols = [start_position_at_step[1], next_optimal[1]]
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
                noise_mult = 0.2
                # noise_mult = 0
                row_list = [el + noise_mult*np.random.randn() for el in path_rows]
                row_list_truncated = [el if el < max_height else max_height for el in row_list]
                row_list_truncated = [el if el > min_height else min_height for el in row_list_truncated]
                if j == labelled_index:
                    ax.plot(path_cols, tuple(row_list_truncated),
                            'k--', linewidth=0.25, label='Further trajectory options')
                else:
                    # continue
                    ax.plot(path_cols, tuple(row_list_truncated),
                            'k--', linewidth=0.25)
            # ax.set_title('Result with Optimal Path', fontsize=18)
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

        # Set x-ticks and labels
        ax.set_xticks(x_ticks_positions)
        ax.set_xticklabels(x_ticks_labels)
        # Set y-ticks and labels
        ax.set_yticks(y_ticks_positions)
        ax.set_yticklabels(y_ticks_labels_str)

        fig.savefig(f'{plot_path}mean_earth_{i}.png', bbox_inches='tight')
        fig.savefig(f'{plot_path}mean_earth_{i}.pdf', bbox_inches='tight')

        print(f'Saved step {i}')

        if not saved_legend:
            # Adding the legend outside the plot
            # manually adding a line to the legend
            cbar = plt.colorbar(im, ax=ax, location='bottom')
            ax.plot([0], [0], linestyle=contour_style, color=contour_color,
                     label='Outline of sand+crevasse in the true model')

            legend = ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left')
            legend.get_frame().set_facecolor('lightgray')  # Set the background color to light gray
            fig.savefig(f'{plot_path}legend.png', bbox_inches='tight')
            fig.savefig(f'{plot_path}legend.pdf', bbox_inches='tight')
            saved_legend = True


        # drilled_path.append(checkpoint_at_step['pos'])# note that we compute another one during viosualization

        # plt.show()

        # plot the mean GAN output for the current decision points

main()
