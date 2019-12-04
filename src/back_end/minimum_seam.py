import numpy as np
from numba import jit
from back_end.calc_energy import *

#--------------------------------------------------------#
@jit
def minimum_seam(img, operator = 'Sobel'):
    r, c, _ = img.shape
    energy_map = calc_energy(img, operator)

    M = energy_map.copy() #Deep copy so we don't have ties between arrays
    #instantiate cost matrix. This is the same shape as the images energy map
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r): #ignore top row. Cost matrix values for top row are = to energy map
        for j in range(0, c):
            # http://cs.brown.edu/courses/cs129/results/proj3/taox/
            #If a neighboring pixel is not available due to the left or right edge,
            #it is simply not used in the minimum of top neighbors calculation.
            if j == 0:
                '''
                [5,5,9
                 10,1,11,
                 5,3,1]

                '''
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack
