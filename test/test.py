import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf, tensorflow.keras as keras
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.models import Model
import cv2
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# Loads all images found and trains given model
def load_images_and_train_model(count, model):
    with tf.device('/GPU:0'):
        dfY = pd.read_csv("../data/train_labels.csv")[:count * 100] # train on count number of image chains (each chain has 100 images)
        # Generate path for each image (necessary for image generator)
        dfY["path"] = [f"{id}/{i:03d}.png" for id, i in zip(dfY["chain_id"], dfY["i"])]
        dfY.drop(["chain_id", "i"], axis=1, inplace=True)
        train_datagen = ImageDataGenerator()
        
        # Create generator to read images from directories
        train_generator = train_datagen.flow_from_dataframe(dataframe=dfY,
            directory="../data/images",
            x_col="path",
            y_col=["x", "y", "z", "qw", "qx", "qy", "qz"],
            target_size=(640, 512), # Downscale
            color_mode='grayscale',
            class_mode="raw")
        
        model.fit(train_generator, epochs=2)

    return model


# x_train = dfX[:9]
# y_train = dfY[:9]
# x_test = dfX[-9:]
# y_test = dfY[-9:]
# print(dfY)
# print(dfX.shape)
# dfY.fillna(value=0, inplace=True)
# plt.imshow(dfX.tolist(), cmap="gray") # plot 1st image's 2nd feature map
# plt.show()


input = Input(shape=(640,512,1))
conv = Conv2D(4, 4, activation='relu', padding="same", input_shape=[640, 512, 1])(input)
#BatchNormalization()(conv)
pooling = MaxPooling2D(2)(conv)
conv2 = Conv2D(3, 2, activation='relu', padding='same')(pooling)
flat = Flatten()(conv2)
hidden = Dense(16, activation="relu")(flat)
# Multiple outputs: https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
x = Dense(1)(hidden)
y = Dense(1)(hidden)
z = Dense(1)(hidden)
qw = Dense(1)(hidden)
qx = Dense(1)(hidden)
qy = Dense(1)(hidden)
qz = Dense(1)(hidden)

model = Model(inputs=input, outputs=[x,y,z,qw,qx,qy,qz])

model.summary()
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))
load_images_and_train_model(2, model)