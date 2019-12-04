import numpy as np
from scipy.ndimage.filters import convolve

#Dynamic Programming Stuff
#--------------------------------------------------------#
#--------------------------------------------------------#
#--------------------------------------------------------#
def calc_energy(img, operator):

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
    # Stack Filter for all dims


    filter_dx = np.stack([dx] * 3, axis=2)
    filter_dy = np.stack([dy] * 3, axis=2)


    #img to array
    img = img.astype('float32')

    #Convolve filter over image channels
    #http://cs.brown.edu/courses/cs129/results/proj3/taox/
    #For each color channel, the energy is calculated by adding the
    #absolute value of the gradient in the x direction to the absolute value of the gradient in the y direction.
    
    convolved = np.absolute(convolve(img, filter_dx)) + np.absolute(
        convolve(img, filter_dy))

    energy = convolved.sum(axis=2)
    #Energy returns image gradient with filter convolved over each dim
    return energy
