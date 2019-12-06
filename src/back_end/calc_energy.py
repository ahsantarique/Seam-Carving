import numpy as np
from scipy.ndimage.filters import convolve

#Dynamic Programming Stuff
#--------------------------------------------------------#
#--------------------------------------------------------#
#--------------------------------------------------------#

def get_filter(operator):

    if operator == 'Sobel':
        dx = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])
        dy = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])

    if operator == 'Sobel_Feldman':
        dx = np.array([
            [3.0, 0.0, -3.0],
            [10.0, 0.0, -10.0],
            [3.0, 0.0, -3.0],
        ])

        dy = np.array([
            [3.0, 10.0, 3.0],
            [0.0, 0.0, 0.0],
            [-3.0, -10.0, -3.0],
        ])


    if operator == 'Scharr':
        dx = np.array([
            [47.0, 0.0, -47.0],
            [162.0, 0.0, -162.0],
            [47.0, 0.0, -47.0],
        ])

        dy = np.array([
            [47.0, 162.0, 47.0],
            [0.0, 0.0, 0.0],
            [-47.0, -162.0, -47.0],
        ])

    return dx, dy


def calc_energy(img, operator):

    dx, dy = get_filter(operator)

    filter_dx = np.stack([dx] * 3, axis=2)
    filter_dy = np.stack([dy] * 3, axis=2)


    img = img.astype('float32')

  
    convolved = np.absolute(convolve(img, filter_dx)) + np.absolute(
        convolve(img, filter_dy))

    energy = convolved.sum(axis=2)

    return energy
