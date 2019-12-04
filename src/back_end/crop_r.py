#--------------------------------------------------------#
#Handle Row-Wise Carving
#--------------------------------------------------------#
def crop_r(img, scale_r , save_progress = 10, operator = 'Sobel'):
    '''
    Backbone for main carve method.
    Uses crop_c under the hood
    '''
    #Meta Heuristic
    img = np.rot90(img, 1, (0, 1)) #Rotate 90degrees
    img = crop_c(img, scale_r, save_progress = save_progress, rotation=True, operator= operator) #Carve
    img = np.rot90(img, 3, (0, 1)) #Rotate Back
    return img

