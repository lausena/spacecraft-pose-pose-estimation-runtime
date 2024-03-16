import numpy as np
import cv2

# https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
# https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/


# def find_cm(im):

#     m = np.sum(np.asarray(im), -1) < 255*3
#     m = m / np.sum(np.sum(m))

#     dx = np.sum(m, 0) # there is a 0 here instead of the 1
#     dy = np.sum(m, 1) # as np.asarray switches the axes, because
#                     # in matrices the vertical axis is the main
#                     # one, while in images the horizontal one is
#                     # the first
    
#     X, Y = im.shape[1], im.shape[0]
#     cx = np.sum(dx * np.arange(X))
#     cy = np.sum(dy * np.arange(Y))

#     return int(cx), int(cy)

def find_cm(img):

    # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    
    # calculate moments of binary image
    M = cv2.moments(thresh)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

# def find_cm2(im):

#     m = np.sum(np.asarray(im), -1) < 255*3
#     m = m / np.sum(np.sum(m))

#     dx = np.sum(m, 0) # there is a 0 here instead of the 1
#     dy = np.sum(m, 1) # as np.asarray switches the axes, because
#                     # in matrices the vertical axis is the main
#                     # one, while in images the horizontal one is
#                     # the first
    
#     return dx, dy
