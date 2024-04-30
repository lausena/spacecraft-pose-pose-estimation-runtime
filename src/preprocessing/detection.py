
import cv2
import numpy as np

def detect_black_image(img):
    perc_threshold = 0.00001
    number_of_non_black_pix = np.sum(img > 10) 
    if number_of_non_black_pix < perc_threshold*img.size:
        return True
    return False

# May use this later
def detect_blur(gray_im, threshold):

    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray_im, cv2.CV_64F)

    # Calculate maximum intensity and variance
    _, max_val, _, _ = cv2.minMaxLoc(gray_im)

    laplacian_variance = laplacian.var()

    blurry = False
    # Check blur condition based on variance of Laplacian image
    if laplacian_variance < threshold:
        blurry = True

    return blurry

# May use this later
def detect_bright_spots(gray_im):

    # Apply binary thresholding for bright spot detection
    _, binary_image = cv2.threshold(gray_im, 200, 255, cv2.THRESH_BINARY)
    binary_variance = binary_image.var()

    # Check bright spot condition based on variance of binary image
    if 5000 < binary_variance < 8500:
        bright_spots = True

    return bright_spots