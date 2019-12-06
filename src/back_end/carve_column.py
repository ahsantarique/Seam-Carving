from numba import jit
from back_end.minimum_seam import *

#--------------------------------------------------------#
@jit
def carve_column(img, operator = 'Sobel'):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img, operator)


    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]


    mask = np.stack([mask] * 3, axis=2)

    img = img[mask].reshape((r, c - 1, 3))

    return img
