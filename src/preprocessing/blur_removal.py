
import os
import cv2
import numpy as np
from imutils import paths

# https://medium.com/@sahilutekar.su/detecting-blur-and-bright-spots-in-images-using-python-and-opencv-6bab8ce75404
# https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv


ADD_TEXT = True
blur_threshold = 50.0

def detect_and_remove_blur(im):

    # Convert image to grayscale
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    blurry = detect_blur_and_bright_spot(gray_im, blur_threshold)

    modified = False
    if blurry:
        im = remove_blur(im)
        modified = True

    return im, modified




def add_blurry_text(im):

    # Initialize result variables
    blur_text = "Blurry"
    cv2.putText(im, blur_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # bright_spot_text = "No Bright Spot"
    # cv2.putText(im, bright_spot_text, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    return im


def create_output_filepath(image_path, output_dir):
    output_filename = image_path.split("\\")[-1]
    output_filename = output_filename.split(".")[0]
    output_file = os.path.join(output_dir, output_filename)
    output_file = output_file + ".jpg"
    return output_file