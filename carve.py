#!/usr/bin/env python
# coding: utf-8

# In[19]:


import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from PIL import Image
import PySimpleGUI as sg
import os, fnmatch, glob
import imageio
from matplotlib.image import imread
import PySimpleGUI as sg
import os, fnmatch
import glob
from tqdm import trange
from numba import jit
import shutil
import os
from os.path import splitext
import warnings
warnings.filterwarnings('ignore')
import time

import json


#!pip install pysimplegui

#Did not touch the below 3 methods
#Dynamic Programming Stuff
#--------------------------------------------------------#
#--------------------------------------------------------#
#--------------------------------------------------------#
def calc_energy(img):
    #NEED: Param for filter type. Believe this is using Sobel.
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
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(
        convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map


#--------------------------------------------------------#
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


#--------------------------------------------------------#
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


#--------------------------------------------------------#
#--------------------------------------------------------#
#--------------------------------------------------------#

#--------------------------------------------------------#
#Create Gif From all Temp Images Stored during Carving
#--------------------------------------------------------#
def create_gif():               #Create Gif from images partitioned through carving
    files = os.listdir("temp/")
    files = ["temp/" + file for file in files]
    files.sort(key=lambda x: os.path.getmtime(x)) #Sort by Date Deposited. Otherwise Bad Ordering on GIF
    images = []
    for filename in files:
        images.append(imageio.imread(filename))
    imageio.mimsave('gif/movie.gif', images, duration=.1) #writeout GIF


#--------------------------------------------------------#
#Progressively Carve and Store Images at checkpoints
#--------------------------------------------------------#
def crop_c(img, scale_c, save_progress = 10, rotation=False):
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

        img = carve_column(img)
                                #Store image after every k cols/rows carved for gif generation
        if i % save_progress == 0:
                                #Handle Rotation Partitions for Row-wise Carving
            if rotation:
                imageio.imsave('temp/{}'.format('a' + str(i) + '.png'),
                               np.rot90(img, 3, (0, 1)))
            else:
                imageio.imsave('temp/{}'.format('a' + str(i) + '.png'), img)

                                #Create Gif from images partitioned through carving
    create_gif()
    return img


#--------------------------------------------------------#
#Handle Row-Wise Carving
#--------------------------------------------------------#
def crop_r(img, scale_r , save_progress = 10):
    '''
    Backbone for main carve method.
    Uses crop_c under the hood

    '''
    img = np.rot90(img, 1, (0, 1)) #Rotate 90degrees
    img = crop_c(img, scale_r, save_progress = save_progress, rotation=True) #Carve
    img = np.rot90(img, 3, (0, 1)) #Rotate Back
    return img


#--------------------------------------------------------#
#Main Carving Method
#--------------------------------------------------------#
def carve(img, dim, output, scale=0.5 , save_progress = 10):
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
        new = crop_c(img, scale_c=scale, save_progress = save_progress)
        imageio.imsave(output, new)

    if dim == 'Row':
        new = crop_r(img, scale_r=scale, save_progress = save_progress)
        imageio.imsave(output, new)



#Helpers
#--------------------------------------------------------#
def transform_images():
    '''
    returns: None
    Runs prior to GUI launch.
    PySimpleGUI only takes pngs as input. Need to transform all jpgs apriori
    '''
    for infile in glob.glob("*.jpg"):
        print(infile)
        file, ext = os.path.splitext(infile)
        print(file, ext)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(directory + file + ".png", "PNG")
        os.remove(infile)
#--------------------------------------------------------#
def create_new_path(path):
    '''
    path: str; will be passed image path from dictionary returned by GUI form fill
    returns: Temp path to store final resized image

    '''
    path = os.path.basename(path)
    file_name, extension = splitext(path)
    return file_name + '_carved' + extension


#--------------------------------------------------------#


# In[24]:


try:
    #Some Cleaning of Dir + Fetching
    #--------------------------------------------------------#

    directory = os.getcwd() + '\\' +'images' #Get CWD
    transform_images()  #Transform images
    #Get list of images in CWD
    files = [
        item for sublist in
        [glob.glob(directory + ext) for ext in ["/*.png", "/*.jpg", "/*.gif"]]
        for item in sublist
    ]
    #--------------------------------------------------------#

    #Create Temp Storage
    #--------------------------------------------------------#

    try:
        os.makedirs('temp')  #This is where progress images are stored
        os.makedirs('gif')  #This is where Gif will be stored
        os.makedirs('carved')
    except:
        #Delete Folders if already in DIR
        shutil.rmtree('temp/')
        shutil.rmtree('gif/')
        shutil.rmtree('carved/')
        os.makedirs('temp')
        os.makedirs('gif')
        os.makedirs('carved')
#--------------------------------------------------------#

#Window One Design
#--------------------------------------------------------#
    sg.ChangeLookAndFeel('Material1')

    # ------ Menu Definition ------ #
    menu_def = [['File', ['Exit']]]

    # ------ Column Definition ------ #
    column1 = [
        [
            sg.Text('Column 1',
                    background_color='#F7F3EC',
                    justification='center',
                    size=(10, 1))
        ],
        [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],
        [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],
        [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]
    ]

    layout = [
        [sg.Menu(menu_def, tearoff=True)],
        [
            sg.Text('Seam Carving - content-aware image resizing',
                    size=(40, 1),
                    justification='center',
                    font=("Helvetica", 24),
                    relief=sg.RELIEF_RIDGE)
        ],

         # ------ Image Selector ------ #
        [
            sg.Text(
                'Choose An Image',
                size=(20, 1),
                auto_size_text=True,
                justification='center',
                tooltip=
                'Image Selection. Gui can only display .png files. jpgs run through conversion apriori.'
            ),
            sg.InputCombo(([i for i in files]),
                          default_value=files[0],
                          size=(80, 1))
        ],
        [
            sg.Frame(layout=[
                # ------ Dimension Selector ------ #
                [
                    sg.Text('Choose A Dim To Carve Along',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Row or Column Carving'),
                    sg.InputCombo(('Column', 'Row'),
                                  default_value='Column',
                                  size=(20, 1))
                ],
                # ------ Filter Selector ------ #

                [
                    sg.Text('Choose A Filter To Use',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Filter Selection'),
                    sg.InputCombo(('Sobel', 'Kalman'),
                                  default_value='Sobel',
                                  size=(20, 1))
                ],
                # ------ Rescale Size Selector ------ #
                [
                    sg.Text('Rescaling Factor',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Filter Selection'),
                    sg.Slider(range=(1, 100),
                              orientation='h',
                              size=(34, 20),
                              default_value=50)
                ],
                # ------ Save Progress Slider ------ #
                [
                    sg.Text('Save Every K Seams',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Filter Selection'),
                    sg.Slider(range=(1, 20),
                              orientation='h',
                              size=(34, 20),
                              default_value=10)
                ],
            ],
                     title='Options',
                     title_color='red',
                     relief=sg.RELIEF_SUNKEN,
                     tooltip='Set Parameters to Feed into SC algo')
        ],
        [sg.Text('_' * 80)],
        [sg.Button('Launch'),
         sg.Cancel()],
    ]

    window = sg.Window('Seam Carving',
                       layout,
                       default_element_size=(40, 1),
                       grab_anywhere=False)

    window2_active = False

    event, values = window.Read()

    if event == 'Exit':
        window.Close()
#--------------------------------------------------------#
#Where the magic happens
#--------------------------------------------------------#

    if not window2_active and event == 'Launch':
        #Dump Formfill Data b4 overwritten & Reload
        #--------------------------------------------------------#
        with open('file.txt', 'w') as file:
            file.write(json.dumps(values))
        vals = json.load(open("file.txt"))
        '''
        vals structure (Exported Forms):
        [1]: Main Folder. Change to image folder
        [2]: Image Path. Change to show subpath per [1]
        [3]: Row/Column parameter
        [4]: Filter Type. To do. Defaults to Sobel?
        [5]: Rescale Size. Resize to [0,1] range.
        [6]: How often to checkpoint an image during carving. Defaults to 10. >10 is slow on gif creation

        '''
        #Run Main Carving Method
        #--------------------------------------------------------#

        output = 'carved/' + create_new_path(
            vals['2'])  #specify carved output location

        carve(vals['2'], dim=vals['3'], output=output,
              scale=vals['5'] / 100, save_progress= vals['6']) #run seam carving

        window2_active = True

        #Layout for window 2 specified. Show Original and Carved image side by side.
        #--------------------------------------------------------#
        layout2 = [
            #[sg.PopupAnimated('gif\movie.gif')],
            [
                sg.Text('Original Image'),
                sg.Image(r"{}".format(vals['2'])),
                sg.Image(r"{}".format(output)),
                sg.Text('Carved Image')
            ],
            [sg.Button('Exit')],
        ]

        #Popup Window Showing Carve Progression. Max Time on screen is fixed to 30 seconds before disappearing
        #--------------------------------------------------------#
        timeout = time.time() + 30  #30 second popup limit
        frames = len(os.listdir("temp/"))  #Count of frames in Gif
        while True:
            if time.time() < timeout:
                sg.PopupAnimated(image_source=r"gif\movie.gif",
                                 time_between_frames=(30 / frames * 1000) / 2,
                                 message="Showing Carve Progression")
            else:
                break

        sg.PopupAnimated(None)
        #Launch Second Window
        #--------------------------------------------------------#

        window2 = sg.Window('Window 2', resizable=True).Layout(layout2)

        #EXIT Second Window
        #--------------------------------------------------------#
        if window2_active:
            ev2, vals2 = window2.read()
            if ev2 is None or ev2 == 'Exit':
                window2_active = False
                window2.close()

    #EXIT Base Window
    #--------------------------------------------------------#
    window.Close()

except:
    sg.Popup('Something went wrong')
    #os.remove(output)
    window.Close()


# In[ ]:
