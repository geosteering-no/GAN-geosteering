import numpy as np
import math

def get_resistivity_default(pixel: np.ndarray):
    return get_resistivity_6_chanel_simple(pixel)


def _get_resistivity_dumb(pixel: np.ndarray):
    return pixel[1]+2


def get_resistivity_archy(porosity, water_saturation=0.7, water_resistivity=0.23, m=2, n=2, tortuosity=0.8):
    """

    :param porosity:
    :param water_saturation:
    :param water_resistivity:
    :param m:
    :param n:
    :param tortuosity:
    :return:
    """
    return (porosity**(-m)) * (water_saturation**(-n)) * tortuosity * water_resistivity


def get_resistivity_6_chanel_based_on_facies(pixel: np.ndarray, clip_1_low=0.9, clip_1_high=4.1, clip_at=14.14, clip_2_low=50.0, clip_2_high=220.0):
    """

    :param pixel: an array of 6 channels
    0 probability of Shale
    1 probability of Sand
    2 probability of Crevasse
    3 porosity in Shale
    4 porosity in Sand
    5 porosity in Crevasse
    :return:
    """
    pixel_facies = pixel[0:3]
    facies_ind = np.argmax(pixel_facies)
    if facies_ind == 0:
        porosity = 0.02
        w_saturation = 0.5
    elif facies_ind == 2:
        porosity = 0.14
        w_saturation = 0.1
    else:
        porosity = 0.25
        w_saturation = 0.1
    resist = get_resistivity_archy(porosity, water_saturation=w_saturation)
    print("Archies resistivity", resist)
    if resist <= clip_at:
        resist = max(resist, clip_1_low)
        resist = min(resist, clip_1_high)
    else:
        resist = max(resist, clip_2_low)
        resist = min(resist, clip_2_high)
    if math.isnan(resist):
        print(pixel)
    return resist

def get_resistivity_6_chanel_simple(pixel: np.ndarray):
    """
    :param pixel: an array of 6 channels
    0 probability of Shale
    1 probability of Sand
    2 probability of Crevasse
    3 porosity in Shale
    4 porosity in Sand
    5 porosity in Crevasse
    :return:
    """
    pixel_facies = pixel[0:3]
    facies_ind = np.argmax(pixel_facies)
    if facies_ind == 0:
        resist = 4.
    elif facies_ind == 2:
        resist = 55.
    else:
        resist = 171.

    return resist

if __name__ == "__main__":
    pixel = np.array([1, 0, 0, 0, 0, 0])
    resist = get_resistivity_6_chanel_simple(pixel)
    print(pixel, resist)

    pixel = np.array([0, 1, 0, 0, 0, 0])
    resist = get_resistivity_6_chanel_simple(pixel)
    print(pixel, resist)

    pixel = np.array([0, 0, 1, 0, 0, 0])
    resist = get_resistivity_6_chanel_simple(pixel)
    print(pixel, resist)
