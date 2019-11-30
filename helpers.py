import glob
from PIL import Image
import os
from os.path import splitext
import shutil
import imageio
import matplotlib.pylab as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters


#Helpers
#--------------------------------------------------------#
def transform_images():
    '''
    returns: None
    Runs prior to GUI launch.
    PySimpleGUI only takes pngs as input. Need to transform all jpgs apriori
    '''
    for infile in glob.glob("images/*.jpg"):
        print(infile)
        file, ext = os.path.splitext(infile)
        print(file, ext)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(file + ".png", "PNG")
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


def save_energy_map(img):
    im = rgb2gray(imageio.imread(img)) # RGB image to gray scale
    plt.gray()
    plt.figure(figsize=(20,10))
    plt.subplot(221)
    plt.imshow(im)
    plt.title('original', size=10)
    plt.subplot(222)
    edges = filters.sobel(im)
    plt.imshow(edges)
    plt.title('sobel', size=10)
    plt.savefig('energy/energymap.png', dpi = 100, bbox_inches='tight')
