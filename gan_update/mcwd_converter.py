import numpy as np
from resitivity import get_resistivity_default


def convert_to_mcwd_input(
        pixel_input: np.ndarray,
        middle_index=15,
        convert_to_resistivity=get_resistivity_default,
        angle_deg=90,
        cell_height=0.5,
        max_bounds=6):
    """

    :param pixel_input: np_array ind_0: pixel channels, ind_1: pos
    :param middle_index:
    :param convert_to_resistivity:
    :param angle_deg:
    :param cell_height:
    :param max_bounds:
    :return:
    """
    # TODO: Look at this again. Maybe 3 is not the best choice.
    middle_index += 3
    tmp = [convert_to_resistivity(pix) for pix in pixel_input.T]
    new_tmp =[tmp[0], tmp[0], tmp[0]] + tmp + [tmp[-1], tmp[-1],tmp[-1]]
    single_column = np.array(new_tmp)
    diff_vec = np.diff(single_column)
    ind_vec = (np.arange(0, len(diff_vec)) - middle_index - 0.5) * cell_height
    # TODO split in the middle
    weight_diff = np.abs(diff_vec) / ind_vec
    half_bounds = max_bounds // 2
    top_inds_all = (weight_diff[:middle_index+1]).argsort()
    top_inds = top_inds_all[:half_bounds]
    bottom_inds_all = (-weight_diff[middle_index+1:]).argsort()+middle_index+1
    bottom_inds = bottom_inds_all[:half_bounds]
    sorted_inds = np.concatenate((top_inds, bottom_inds))
    sorted_inds = np.sort(sorted_inds)
    # resistivity resistivity boundary etc
    # angle angle
    # first part r
    result = [single_column[sorted_inds[0]], single_column[sorted_inds[0]]]
    for i in range(len(sorted_inds)-1):
        #depth
        result.append(ind_vec[sorted_inds[i]])

        start = sorted_inds[i]+1
        end_excl = sorted_inds[i+1]+1
        r_average = np.average(single_column[start:end_excl])
        # next part r
        result.append(r_average)
        result.append(r_average)

    #depth
    last_ind = len(sorted_inds)-1
    result.append(ind_vec[sorted_inds[last_ind]])

    # final resistivities
    r_final = single_column[sorted_inds[last_ind] + 1]
    result.append(r_final)
    result.append(r_final)

    #angles
    result.append(angle_deg)
    result.append(angle_deg)

    return np.array(result)








