
import sys

import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from PIL import Image
try:
    import PySimpleGUI as sg
except:
    import PySimpleGUI as sg
import os, fnmatch, glob
import imageio
import os, fnmatch
import glob
from tqdm import trange
try:
    from numba import jit
except:
    from numba import jit
from os.path import splitext
import json

#!pip install pysimplegui

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

@jit
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

@jit
def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c): # use range if you don't want to use tqdm
        img = carve_column(img)
        sg.OneLineProgressMeter('Stay Tuned.. Carving Image', i+1, c - new_c, 'key')
    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img



def carve(img, dim, output, scale = 0.5):
    img = imageio.imread(img)
    if dim == 'Column':
        new = crop_c(img, scale_c = scale)
        imageio.imsave(output, new)

    if dim =='Row':
        new = crop_r(img, scale_r = scale)
        imageio.imsave(output, new)


def create_new_path(path):
    path = os.path.basename(path)
    file_name,extension = splitext(path)
    file_name + '_carved' + extension
    return file_name + '_carved' + extension

if __name__ == "__main__":
    directory = os.getcwd() + '\\'

    for infile in glob.glob("*.jpg"):
        print(infile)
        file, ext = os.path.splitext(infile)
        print(file, ext)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(directory + file + ".png", "PNG")
        os.remove(infile)


    files = [item for sublist in [glob.glob(directory + ext) for ext in ["/*.png", "/*.jpg", "/*.gif"]] for item in sublist]

    #Window One Design
    ###############################################
    sg.ChangeLookAndFeel('Material1')

    # ------ Menu Definition ------ #
    menu_def = [['File', ['Exit']]]

    # ------ Column Definition ------ #
    column1 = [[sg.Text('Column 1', background_color='#F7F3EC', justification='center', size=(10, 1))],
               [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],
               [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],
               [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]]

    layout = [
        [
            sg.Menu(menu_def, tearoff=True)],
            [
            sg.Text('Seam Carving - content-aware image resizing',
            size=(40, 1), justification='center', font=("Helvetica", 24), relief=sg.RELIEF_RIDGE)
            ],

            [
            sg.Text('Choose A Folder', size=(20, 1), auto_size_text=True, justification='center',
            tooltip='Directory Selection. Defaults to current Working Directory.'),
            sg.InputText(directory, size=(80, 1)), sg.FolderBrowse()
            ],

            [
            sg.Text('Choose An Image', size=(20, 1), auto_size_text=True, justification='center',
            tooltip='Image Selection. Gui can only display .png files. jpgs run through conversion apriori.'),
            sg.InputCombo(([i for i in files]), default_value=files[0],  size=(80, 1))
            ],


            [
            sg.Frame(layout=
                [
                    [
            sg.Text('Choose A Dim To Carve Along', size=(25, 1), auto_size_text=True, justification='center',

            tooltip='Row or Column Carving'),
            sg.InputCombo(('Column', 'Row'), default_value='Column', size=(20, 1))
                    ],
                    [
            sg.Text('Choose A Filter To Use', size=(25, 1), auto_size_text=True, justification='center',

            tooltip='Filter Selection'),
            sg.InputCombo(('Sobel', 'Kalman'), default_value='Sobel', size=(20, 1))
                    ],

                    [
            sg.Text('Rescaling Factor', size=(25, 1), auto_size_text=True, justification='center',

            tooltip='Filter Selection'),
            sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=50)
                    ],

                ],
            title='Options',title_color='red', relief=sg.RELIEF_SUNKEN, tooltip='Set Parameters to Feed into SC algo')
            ],



        [
            sg.Text('_'  * 80)
        ],

        [
            sg.Button('Launch'), sg.Button('Preview Image'), sg.Cancel()
        ] ,
    ]
    ###############################################


    window = sg.Window('Seam Carving', layout, default_element_size=(40, 1), grab_anywhere=False)

    window2_active = False

    event, values = window.Read()

    if event == 'Exit':
        window.Close()

    if not window2_active and event == 'Launch':

        with open('file.txt', 'w') as file:
            file.write(json.dumps(values))

        vals = json.load(open("file.txt"))

        output = create_new_path(vals['2'])
        carve(vals['2'], dim = vals['3'], output = output, scale = vals['5'] /100 )
        #event3, values3 = window_3.read(timeout=10)

        window2_active = True
        layout2 = [
            #[sg.Text('                         Orginal Image ---------------                            Carved Image')],
            [sg.Text('Original Image'), sg.Image(r"{}".format(vals['2'])),
             sg.Image(r"{}".format(output)), sg.Text('Carved Image')],
            [sg.Button('Exit')],
        ]

        window2 = sg.Window('Window 2',resizable=True).Layout(layout2)


        if window2_active:
            ev2, vals2 = window2.read()
            if ev2 is None or ev2 == 'Exit':
                window2_active  = False
                window2.close()

    os.remove(output)
    window.Close()
