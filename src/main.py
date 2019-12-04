# coding: utf-8

import sys
import numpy as np
from imageio import imread, imwrite
from PIL import Image
import PySimpleGUI as sg
import os, fnmatch, glob
import imageio
from matplotlib.image import imread
import os, fnmatch
import glob
from tqdm import trange
from numba import jit
import os
import warnings
warnings.filterwarnings('ignore')
import time
import json

from util.directory import *
from util.images import *

from back_end.carve import *

from front_end.layouts import *


if __name__ == '__main__':
    try:
        #Some Cleaning of Dir + Fetching
        #--------------------------------------------------------#
        directory = os.getcwd() + '/../' +'images' #Get CWD
        transform_images()  #Transform images
        #Get list of images in CWD
        # files = [
        #     item for sublist in
        #     [glob.glob(directory + ext) for ext in ["/*.png", "/*.jpg", "/*.gif"]]
        #     for item in sublist
        # ]
        #--------------------------------------------------------#
        #Create Temp Storage
        #--------------------------------------------------------#
        create_dirs()
        #--------------------------------------------------------#

        layout = get_browser_layout()
        
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

            print("json vals:", values)
            
            '''
            vals structure (Exported Forms):
            ['Browse']: Image Path.
            [1]: Row/Column parameter
            [2]: Filter Type. To do. Defaults to Sobel?
            [3]: Rescale Size. Resize to [0,1] range.
            [4]: How often to checkpoint an image during carving. Defaults to 10. >10 is slow on gif creation
            '''
            #Run Main Carving Method
            #--------------------------------------------------------#

            image_path = values['Browse']
            dim = values[1]
            operator = values[2]
            scale = values[3]
            save_period = values[4]

            output = '../carved/' + create_new_path(image_path)  #specify carved output location
            carve(image_path, dim=dim, output=output, scale = scale/ 100, save_progress= save_period, operator= operator) #run seam carving

            save_energy_map(image_path)

            #Popup Window Showing Carve Progression. Max Time on screen is fixed to 30 seconds before disappearing
            #--------------------------------------------------------#
            timeout = time.time() + 10  #30 second popup limit
            frames = len(os.listdir("../temp/"))  #Count of frames in Gif

            print("here")

            # while True:
            #     if time.time() < timeout:
            #         sg.PopupAnimated(image_source=r"gif\movie.gif",
            #                          time_between_frames=(30 / frames * 1000) / 6,
            #                          message="Showing Carve Progression. 10 Second Loop")
            #     else:
            #         break

            # print("here2c")

            # sg.PopupAnimated(None)






            window2_active = True
            
            #Layout for window 2 specified. Show Original and Carved image side by side.
            #--------------------------------------------------------#

            print("output:", output)
            layout2 = get_carved_image_layout(image_path = image_path, output = output)

            #Launch Second Window
            #--------------------------------------------------------#
            window2 = sg.Window('Results', resizable=True).Layout(layout2)

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
        os.remove(output)
        window.Close()
