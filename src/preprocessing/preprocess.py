
import cv2
import math
import numpy as np

from preprocessing.kernel_functions import apply_kernel
from preprocessing.center_of_mass import find_cm

# https://medium.com/curious-manava/center-crop-and-scaling-in-opencv-using-python-279c1bb77c74

ADD_CM_Text = False

kernel = "sharpen"

image_crop_x = .75
image_crop_y = .50

BLACK = [0, 0, 0]
PAD_VALUE = 1000

def preprocess_image(im):
    
    # Sharpen the image using sharpening kernel
    kernel_image = apply_kernel(im, kernel)

    # Find the center of mass of the spacecraft
    cX, cY = find_cm(kernel_image)

    kernel_image = cv2.copyMakeBorder(kernel_image, PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE, cv2.BORDER_CONSTANT,value=BLACK)
    
    # Shift center of mass by the pad value
    cX, cY = cX + PAD_VALUE, cY + PAD_VALUE

    im_center_crop = crop_center(kernel_image, cX, cY)

    return im_center_crop


def crop_center(im, cX, cY):

    y_max, x_max, _ = im.shape

    num_pixels_x = math.floor(image_crop_x*(x_max/2))
    num_pixels_y = math.floor(image_crop_y*(y_max/2))

    x_pixel_lb = cX - num_pixels_x
    x_pixel_ub = cX + num_pixels_x

    y_pixes_lb = cY - num_pixels_y
    y_pixes_ub = cY + num_pixels_y

    im_center_crop = im[y_pixes_lb:y_pixes_ub, x_pixel_lb:x_pixel_ub]

    return im_center_crop








# def detect_black_image(img):
#     number_of_non_black_pix = np.sum(img > 5) 
#     if number_of_non_black_pix < 0.00005*img.size:
#         return True
#     return False



# def center_crop(img, dim):
# 	"""Returns center cropped image
# 	Args:
# 	img: image to be center cropped
# 	dim: dimensions (width, height) to be cropped
# 	"""
# 	width, height = img.shape[1], img.shape[0]

# 	# process crop width and height for max available dimension
# 	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
# 	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
# 	mid_x, mid_y = int(width/2), int(height/2)
# 	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
# 	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
# 	return crop_img

# def scale_image(img, factor=1):
# 	"""Returns resize image by scale factor.
# 	This helps to retain resolution ratio while resizing.
# 	Args:
# 	img: image to be scaled
# 	factor: scale factor to resize
# 	"""
# 	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))



# # Defining a function
# def crop_center(im, cx, cy):

#     im_shape = im.shape
#     borderType = cv2.BORDER_CONSTANT

    
#     cv2.copyMakeBorder(im, top, bottom, left, right, borderType)

#     x, y = im.shape[1], im.shape[0]
#     startx = x // 2 - (cx // 2)
#     starty = y // 2 - (cy // 2)
#     return im[starty:starty+cy,startx:startx+cx]
