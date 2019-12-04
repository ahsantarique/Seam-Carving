import glob
import os
import shutil

from os.path import splitext

#--------------------------------------------------------#
def create_new_path(path):
    '''
    path: str; will be passed image path from dictionary returned by GUI form fill
    returns: Temp path to store final resized image

    '''
    path = os.path.basename(path)
    file_name, extension = splitext(path)
    return file_name + '_carved' + extension


def create_dirs():
    try:
        os.makedirs('temp')  #This is where progress images are stored
        os.makedirs('gif')  #This is where Gif will be stored
        os.makedirs('carved')
        os.makedirs('energy')
    except:
        #Delete Folders if already in DIR
        shutil.rmtree('temp/')
        shutil.rmtree('gif/')
        shutil.rmtree('carved/')
        shutil.rmtree('energy/')
        os.makedirs('temp')
        os.makedirs('gif')
        os.makedirs('carved')
        os.makedirs('energy')
