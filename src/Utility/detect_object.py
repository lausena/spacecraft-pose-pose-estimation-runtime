import cv2
import numpy as np

# https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
# https://stackoverflow.com/questions/48129595/how-can-i-detect-an-object-in-image-frame-using-opencv

# def detect(img_rgb):
    
#     gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#     # binarize the image
#     ret, bw = cv2.threshold(gray, 128, 255, 
#     cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#     # find connected components
#     connectivity = 4
#     nb_components, output, stats, centroids = 
#     cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
#     sizes = stats[1:, -1]; nb_components = nb_components - 1
#     min_size = 250 #threshhold value for objects in scene
#     img2 = np.zeros((img.shape), np.uint8)
#     for i in range(0, nb_components+1):
#         # use if sizes[i] >= min_size: to identify your objects
#         color = np.random.randint(255,size=3)
#         # draw the bounding rectangele around each object
#         cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
#         img2[output == i + 1] = color
       

#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#     # Use minSize because for not 
#     # bothering with extra-small 
#     # dots that would look like STOP signs
#     stop_data = cv2.CascadeClassifier('stop_data.xml')

#     found = stop_data.detectMultiScale(img_gray, 
#                                     minSize =(20, 20))
    

#     return found

def draw_rects_around_objects(img_rgb, found):
    # Don't do anything if there's 
    # no sign
    amount_found = len(found)
    if amount_found != 0:
        # There may be more than one
        # sign in the image
        for rectangle in found:
            draw_rectangles(img_rgb, rectangle)

def draw_rectangles(img_rgb, rectangle):
    (x, y, width, height) = rectangle
    # We draw a green rectangle around
    # every recognized sign
    cv2.rectangle(img_rgb, (x, y), 
                (x + height, y + width), 
                (0, 255, 0), 5)