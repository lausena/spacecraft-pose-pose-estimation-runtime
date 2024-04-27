import numpy as np
import math
import tensorflow as tf
from PIL import Image

# https://stackoverflow.com/questions/63827339/how-to-build-a-custom-data-generator-for-keras-tf-keras-where-x-images-are-being
class CustomDataGenerator(tf.keras.utils.Sequence):

    ''' Custom DataGenerator to load img 
    
    Arguments:
        data_frame = pandas data frame in filenames and labels format
        batch_size = divide data in batches
        shuffle = shuffle data before loading
        img_shape = image shape in (h, w, d) format
        augmentation = data augmentation to make model rebust to overfitting
    
    Output:
        Img: numpy array of image
        label : output label for image
    '''
    
    def __init__(self, data_frame, y_labels, batch_size=10, img_shape=None, augmentation=True, num_classes=None):
        self.data_frame = data_frame
        self.labels = y_labels
        self.train_len = len(data_frame)
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_classes = num_classes
        print(f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        ''' return total number of batches '''
        #self.data_frame = shuffle(self.data_frame)
        return np.int32(math.ceil(self.train_len/self.batch_size))

    def on_epoch_end(self):
        ''' shuffle data after every epoch '''
        # fix on epoch end it's not working, adding shuffle in len for alternative
        pass
    
    def __data_augmentation(self, img):
        ''' function for apply some data augmentation '''
        img = tf.image.resize(img, 512, 640)
        img = tf.image.random_flip_up_down(img)
        return img

    def __get_image(self, file_id):
        """ open image with file_id path and apply data augmentation """
        img = np.asarray(Image.open(file_id))
        img = np.resize(img, self.img_shape)
        img = self.__data_augmentation(img)

        return img

    def __get_label(self, label_id):
        """ uncomment the below line to convert label into categorical format """
        return label_id

    def __getitem__(self, idx):
        
        # Create Features
        
        ## Paths
        paths = self.data_frame["Image_Path"][idx * self.batch_size:(idx + 1) * self.batch_size]
        base_paths = self.data_frame["Base_Image_Path"][idx * self.batch_size:(idx + 1) * self.batch_size]

        ## Images
        image = [tf.image.resize(tf.image.decode_png(tf.io.read_file(str(path)), channels=3), (512,640)) for path in paths]
        base_image = [tf.image.resize(tf.image.decode_png(tf.io.read_file(str(path)), channels=3), (512,640)) for path in base_paths]

        ## Range
        all_range = self.data_frame['Range']
        ranges = all_range[idx * self.batch_size:(idx + 1) * self.batch_size]

        #feature_list = [np.asarray(base_image), np.asarray(image), ranges]
        feature_list = [np.asarray(base_image), np.asarray(image)]

        # Create Y Labels
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_labels = np.asarray(y)

        return feature_list, y

