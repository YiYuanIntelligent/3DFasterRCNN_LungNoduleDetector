import cv2
import numpy as np
from PIL import Image
from scipy import misc
import os
import sys
import cv2
from skimage import measure, morphology, segmentation


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image



#input1 =  '../data/train_unet_64_64_norm/img/images_0000_0914.npy'
#input2 = '../data/train_unet_64_64_norm/mask/masks_0000_0914.npy' 

#input = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_clean.npy'
#input = 'test_clean.npy'
input = '../data_test/LKDS-01343_clean.npy'
#input = '../../../../data/val_npy/LKDS-00032.mhd.npy'
#input = '/new_disk_1/tianchi/data/val_lungSeg_npy/LKDS-00993.mhd.npy_clean.npy'

#input = 'lung.npy'

z = 210
x = 373
y = 258

img = np.load(input)
print img.shape

img =img[0][z]
#img1 = np.load(input1)[32]
#img2 = np.load(input2)[32]

#img = np.load(input)

print img[img!=0]


#print img.shape
#img[img<-1024] = 0

#img = normalize(img)
#print img
print img.shape
misc.imsave('img.jpg',img)

chunk_img =img[(x-31):(x+32),(y-31):(y+32)]
print chunk_img,len(chunk_img[chunk_img!=0])
#chunk_img =img[(251-31):(251+33),(379-31):(379+33)]
#img = 255.0 / np.amax(img) * img
#cv2.imwrite('ggg.jpg',img)
misc.imsave('chunk.jpg',chunk_img)



