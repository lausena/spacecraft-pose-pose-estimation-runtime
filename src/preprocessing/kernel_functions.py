import numpy as np
import cv2 


kernels = {
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
            "outline": np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]], np.float32),
            "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32),
            "identity": np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]], np.float32),
            "k2": np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32),
            "k3": np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, 0]], np.float32)
}


def apply_kernel(im, kernel_name):
    # Create the sharpening kernel 
    kernel = kernels[kernel_name]
    im = cv2.filter2D(im, -1, kernel)   
    return im


# def sharpen_image_k2(im):
#     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     im = cv2.filter2D(im, -1, kernel)   
#     return im


# def sharpen_image_k3(im):
#     denominator=3
#     kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32) 
#     kernel /= denominator * kernel
#     im = cv2.filter2D(im, -1, kernel)   
#     pass


# def outline_kernel