import imageio
from back_end.crop_c import *

#--------------------------------------------------------#
#Main Carving Method
#--------------------------------------------------------#
def carve(img, dim, output, scale=0.5 , save_progress = 10, operator = 'Sobel'):
    '''
    Main Method for Seam Carving
    parms:
        img: str; image path. Passed through from GUI.
        dim: str; 'Row' or 'Column'. Passed through from GUI dict.
        output: str; final resting place of carved image.
        scale: float; desired resize. Passed through from GUI.
    returns:
        saves carved image in desired new path.
    '''
    img = imageio.imread(img)
    if dim == 'Column':
        new = crop_c(img, scale_c=scale, save_progress = save_progress, operator = operator)
        imageio.imsave(output, new)

    if dim == 'Row':
        new = crop_r(img, scale_r=scale, save_progress = save_progress, operator = operator)
        imageio.imsave(output, new)
    if dim == 'Both':
        new = crop_c(img, scale_c=scale, save_progress = save_progress, operator = operator)
        new = crop_r(new, scale_r=scale, save_progress = save_progress, operator = operator)
        imageio.imsave(output, new)






