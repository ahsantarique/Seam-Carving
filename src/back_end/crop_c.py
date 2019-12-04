import PySimpleGUI as sg
from back_end.carve_column import *
import imageio
from util.images import *

#--------------------------------------------------------#
#Progressively Carve and Store Images at checkpoints
#--------------------------------------------------------#
def crop_c(img, scale_c, save_progress = 10, rotation=False, operator = 'Sobel'):
    '''
    Backbone for main carve method.
    Parms:
        img: arr (pass through),
        scale_c: float, rescale proportion [0,100]
        rotation: bool, flag on if row-wise carving
    '''
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c):  # use range if you don't want to use tqdm
                                #Need this to share Progress Bar for Seam Carving
        sg.OneLineProgressMeter('Stay Tuned.. Carving Image', i + 1, c - new_c,
                                'key')

        img = carve_column(img, operator = operator)
                                #Store image after every k cols/rows carved for gif generation
        if i % save_progress == 0:
                                #Handle Rotation Partitions for Row-wise Carving
            if rotation:
                imageio.imsave('../temp/{}'.format('a' + str(i) + '.png'),
                               np.rot90(img, 3, (0, 1)))
            else:
                imageio.imsave('../temp/{}'.format('a' + str(i) + '.png'), img)

                                #Create Gif from images partitioned through carving
    create_gif()
    return img

