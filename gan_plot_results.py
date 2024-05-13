import numpy as np
import sys, os
import matplotlib
import matplotlib.pyplot as plt
from input_output import read_config
from GAN import GanLog
home = os.path.expanduser("~") # os independent home

# local load of additional modules.
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/KERNEL'))
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/USER_SERGEY'))
sys.path.append(os.path.join(home,'OneDrive/DISTINGUISH/ECMOR_study/gan-geosteering'))
import mcwd_converter
from vector_to_image import GanEvaluator
from run_model_clean import DnnEvaluatorMcwd
from resitivity import get_resistivity_default
import warnings
# Ignore FutureWarning and UserWarning
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

kd, kf = read_config.read_txt('enkf_test.pipt')
sim = GanLog(kf)

global_extent = [0, 640, -16.25, 15.75]


def get_true():
    numpy_input = np.load('../gan-geosteering/saves/chosen_realization_C1.npz')
    numpy_single = numpy_input['arr_0']
    m_true = numpy_single.copy()
    return m_true


def get_prior():
    prior_mean = np.load('data/mean_field.npz', allow_pickle=True)['arr_0']
    np.random.seed(0)
    # m_prior = np.dot(prior_mean[:, np.newaxis], np.ones((1, 500))) + np.dot(np.linalg.cholesky(0.6 * np.eye(60)),
    #                                                                         np.random.randn(60, 500))
    # m_prior = np.dot(np.linalg.cholesky(0.6 * np.eye(60)), np.random.randn(60, 500))
    m_prior = np.dot(prior_mean[:, np.newaxis], np.ones((1, 500))) \
              + np.dot(np.linalg.cholesky(1.0 * np.eye(60)), np.random.randn(60, 500))
    return m_prior


def get_posterior(filename='final.npz'):
    post_save = np.load(filename, allow_pickle=True)['m']
    return post_save

def get_debug_analysis(filename='debug_analysis_step_1.npz'):
    return np.load(filename, allow_pickle=True)['state'][()]['m']

def save_plot(name):
    plt.savefig('enkf/{}.png'.format(name), bbox_inches='tight', dpi=600)
    plt.savefig('enkf/{}.pdf'.format(name), bbox_inches='tight')


def get_well_direction():
    x = np.arange(0, 121, 30)
    y = x.copy()
    y[0] = 0
    ang = 3. / 180. * np.pi
    for i in range(1,len(x)):
        y[i] = y[i-1] + np.sin(ang) * (x[i] - x[i-1])
        ang += 3./ 180. * np.pi
    return x, y, -y


def plot_resistivity(resistivity_ensemble, label='', plot_realizatoins=False, active_well=False):
    norm = matplotlib.colors.Normalize(vmin=1., vmax=200.0)
    if len(resistivity_ensemble.shape) == 3:
        mean_model = np.mean(resistivity_ensemble, 0)
    else:
        mean_model = resistivity_ensemble
    plt.figure(figsize=(10, 4))
    plt.imshow(mean_model, interpolation='none', norm=norm, cmap='summer', extent=global_extent)
    plt.title('Resistivity mean ({}), ohm m'.format(label))
    plt.colorbar()
    ax = plt.gca()
    ax.axes.set_aspect(8)
    if active_well:
        style_str = 'r'
    else:
        style_str = 'k--'

    plt.plot([-90, 0], [0, 0], style_str, linewidth=2)
    x, y, yy = get_well_direction()
    plt.plot(x, y, 'k--', linewidth=1)
    plt.plot(x, yy, 'k--', linewidth=1)

    save_plot('resisitivity_mean_{}'.format(label))

    if len(resistivity_ensemble.shape) == 3:
        std_model = np.std(resistivity_ensemble, axis=0)
        plt.figure(figsize=(10, 4))
        plt.imshow(std_model, interpolation='none', norm=norm, cmap='summer', extent=global_extent)
        plt.title('Resistivity std ({}), ohm m'.format(label))
        plt.colorbar()
        ax = plt.gca()
        ax.axes.set_aspect(8)
        plt.plot([-90, 0], [0, 0], style_str, linewidth=2)
        x, y, yy = get_well_direction()
        plt.plot(x, y, 'k--', linewidth=1)
        plt.plot(x, yy, 'k--', linewidth=1)
        save_plot('resisitivity_std_{}'.format(label))

    if len(resistivity_ensemble.shape) == 3 and plot_realizatoins:
        for i in range(6):
            plt.figure()
            plt.imshow(resistivity_ensemble[i, :, :], vmin=1., vmax=150., cmap='summer', extent=global_extent)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axes.set_aspect(4)
            plt.plot([-90, 0], [0, 0], style_str, linewidth=2)
            save_plot('resistivity_{}_{}'.format(label, i))
            # plt.title('Facies type')
            plt.close()


def plot_sand_probability(ensemble, label='', plot_realizatoins=False, active_well=False, step=None):
    posterior_earh_models = (ensemble + 1.) / 2.
    if len(ensemble.shape) == 4:
        mean_model = np.mean(posterior_earh_models, 0)
    else:
        mean_model = np.round(ensemble)
    plt.figure(figsize=(10, 4))
    plt.imshow(1. - mean_model[0, :, :], interpolation='none', vmin=0., vmax=1., cmap='tab20b', extent=global_extent)
    if active_well:
        style_str = 'r'
    else:
        style_str = 'k--'

    if step is not None:
        for s in step:
            plt.plot([s*10, s*10], [0, 0], 'r*', linewidth=2)
        plt.plot([0, 640], [0, 0], 'k--', linewidth=2)
        # x, y, yy = get_well_direction()
        # plt.plot(x, y, 'k--', linewidth=1)
        # plt.plot(x, yy, 'k--', linewidth=1)

    plt.title('Probability of sand ({})'.format(label))
    ax = plt.gca()
    ax.axes.set_aspect(8)
    plt.colorbar()
    save_plot('probability_sand_{}'.format(label))

    plt.figure(figsize=(10, 4))
    plt.imshow(mean_model[1, :, :], interpolation='none', vmin=0., vmax=1., cmap='tab20b', extent=global_extent)

    if step is not None:
        for s in step:
            plt.plot([s*10, s*10], [0, 0], 'r*', linewidth=2)
        plt.plot([0, 640], [0, 0], 'k--', linewidth=2)
    # plt.plot([-90, 0], [0, 0], 'k--', linewidth=2)
    # plt.plot([-90 + (90 / 9) * step, -90 + (90 / 9) * (step + 1)], [0, 0], style_str, linewidth=2)
    # x, y, yy = get_well_direction()
    # plt.plot(x, y, 'k--', linewidth=1)
    # plt.plot(x, yy, 'k--', linewidth=1)

    plt.title('Probability of good sand ({})'.format(label))
    ax = plt.gca()
    ax.axes.set_aspect(8)
    plt.colorbar()
    save_plot('probability_good_sand_{}'.format(label))

    if len(ensemble.shape) == 4 and plot_realizatoins:
        index_image = np.argmax(ensemble, 1)
        for i in range(6):
            plt.figure()
            plt.imshow(index_image[i, :, :], cmap='Paired',extent=global_extent)
            # plt.title('Facies type')
            ax = plt.gca()
            ax.axes.set_aspect(4)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            save_plot('facies_{}'.format(i))
            plt.close()

    # plt.figure()
    # plt.imshow(mean_model[2, :, :],interpolation='none',vmin=0.,vmax=1.,cmap='hot')
    # plt.title('Probability of 2')
    # plt.colorbar()


def plot_logs(ensemble, worker,
              use_labels=False,
              indeces=[5, 6, 7, 8, 9, 10, 11, 12],
              names=['Att. 20 kHz', 'Att. 50 kHz', 'Phase 20 kHz', 'Phase 50 kHz', 'Dir. Im. 20 kHz', 'Dir. Im. 50 kHz', 'Dir. Re. 20 kHz', 'Dir. Re. 50 kHz']):
    for i in range(6):
        task = worker.call_sim.remote(input=ensemble[:, i])
        logs = ray.get(task)['res']
        plt.figure(figsize=(10, 2))
        ax = plt.gca()
        if use_labels:
            for j in range(0,len(indeces),2):
                curve_name = names[j]
                plt.plot(np.arange(-80, 1, 10), logs[:, indeces[j]], label=curve_name)
            plt.legend()
            ax.axes.xaxis.set_visible(False)
        else:
            plt.plot(np.arange(-80, 1, 10), logs)

        ax.axes.yaxis.set_visible(False)
        save_plot('logs_{}'.format(i))
        plt.close()


if __name__ == '__main__':
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    true = get_true()
    prior = get_prior()
    posterior = get_posterior('posterior_state_estimate.npz')
    # posterior2 = get_posterior('final2.npz')

    kd, kf = read_config.read_txt('enkf_test.pipt')
    my_gan = GanLog(kf)
    my_gan.gan_evaluator = GanEvaluator(my_gan.file_name, my_gan.vec_size)
    my_gan.mcwd_evaluator = DnnEvaluatorMcwd(trained_model_directory=os.path.join(home,
                                                 'OneDrive/DISTINGUISH/ECMOR_study/deep-borehole-inverse-problem/USER_SERGEY/Adaptive_architecture_2_dataset84599_11746'),
            experiment_name="Adaptive_architecture_2")

    # import gan_animation

    # gan_animation.create_animation(posterior, worker, folder='gif_posterior')
    # gan_animation.create_animation(prior, worker, folder='gif_prior')

    # exit(1)

    # plot_logs(prior, worker)
    #
    # task_true = worker.generate_earth_model.remote(input=true)
    # task_prior = worker.generate_earth_model.remote(input=prior)
    # task_posterior = worker.generate_earth_model.remote(input=posterior)
    # # task_posterior2 = worker.generate_earth_model.remote(input=posterior2)
    #
    # true_earth_model = ray.get(task_true)
    # prior_earth_model = ray.get(task_prior)
    # posterior_earth_model = ray.get(task_posterior)
    # # posterior_earth_model2 = ray.get(task_posterior2)
    # #
    #
    # task_convert = worker.convert_to_resistivity_poor.remote(true_earth_model)
    # true_resistivity = ray.get(task_convert)
    #
    # task_convert = worker.convert_to_resistivity_poor.remote(prior_earth_model)
    # converted_prior = ray.get(task_convert)
    #
    # task_convert = worker.convert_to_resistivity_poor.remote(posterior_earth_model)
    # converted_posterior = ray.get(task_convert)

    # plot_resistivity(true_resistivity, label='true model', active_well=True)
    # plot_resistivity(converted_prior, label='prior', plot_realizatoins=True)
    # plot_resistivity(converted_posterior, label='posterior', plot_realizatoins=True, active_well=True)
    # plt.show()

    ################
    ##################
    # true_earth_model = my_gan.call_sim(state={'m': true}, output_img=True)[2]
    # plot_sand_probability(true_earth_model, label='true_model', active_well=True)
    #
    # prior_earth_model = np.array([my_gan.call_sim(state={'m': prior[:,el]}, output_img=True)[2] for el in range(prior.shape[1])])
    # plot_sand_probability(prior_earth_model, label='prior', plot_realizatoins=True)
    #
    # post = np.array(
    #     [my_gan.call_sim(state={'m': posterior[:, el]}, output_img=True)[2] for el in range(posterior.shape[1])])
    # plot_sand_probability(post, label='ENKF_post', active_well=True, step=range(0,64))
    ##################
    ##################
    for i in range(25,64):
        state = np.load(f'debug_analysis_step_{i}.npz', allow_pickle=True)['state'][()]['m']
        post = np.array(
            [my_gan.call_sim(state={'m': state[:, el]}, output_img=True)[2] for el in range(100)])#range(state.shape[1])])
        plot_sand_probability(post, label=f'ENKF_step_{i}', active_well=True, step=[i])
    # post_step1 = np.array([my_gan.call_sim(state={'m': step1[:,el]}, output_img=True)[2] for el in range(step1.shape[1])])
    # plot_sand_probability(post_step1, label='ENKF_step1', active_well=True,step=0)
    #
    # post_step2 = np.array(
    #     [my_gan.call_sim(state={'m': step2[:, el]}, output_img=True)[2] for el in range(step2.shape[1])])
    # plot_sand_probability(post_step2, label='ENKF_step2', active_well=True,step=1)
    #
    # post_step5 = np.array(
    #     [my_gan.call_sim(state={'m': step5[:, el]}, output_img=True)[2] for el in range(step5.shape[1])])
    # plot_sand_probability(post_step5, label='ENKF_step5', active_well=True,step=4)
    #
    # post_step8 = np.array(
    #     [my_gan.call_sim(state={'m': step8[:, el]}, output_img=True)[2] for el in range(step8.shape[1])])
    # plot_sand_probability(post_step8, label='ENKF_step8', active_well=True,step=7)


    # plot_sand_probability(posterior_earth_model2, label='posterior2')
    # plt.show()
