import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf, tensorflow.keras as keras
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.models import Model
import cv2
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import scripts
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

ToCSV = lambda df, fname: df.round(2).to_csv(f'{fname}.csv')

# Loads all images found and trains given model
def load_images_and_train_model(count, model):
    with tf.device('/GPU:0'):
        dfY = pd.read_csv("../data/train_labels.csv")[:count * 100] # train on count number of image chains (each chain has 100 images)
        # Generate path for each image (necessary for image generator)
        dfY["path"] = [f"{id}/{i:03d}.png" for id, i in zip(dfY["chain_id"], dfY["i"])]
        dfY.drop(["chain_id", "i"], axis=1, inplace=True)
        train_datagen = ImageDataGenerator(validation_split=0.1)
        
        # Create generator to read images from directories
        train_generator = train_datagen.flow_from_dataframe(dataframe=dfY,
            directory="../data/images",
            x_col="path",
            y_col=["x", "y", "z"],
            target_size=(640, 512), # Downscale
            color_mode='grayscale',
            class_mode="raw",
            batch_size=128,
            subset="training")
        
        test_generator = train_datagen.flow_from_dataframe(dataframe=dfY,
            directory="../data/images",
            x_col="path",
            y_col=["x", "y", "z"],
            target_size=(640, 512), # Downscale
            color_mode='grayscale',
            class_mode="raw",
            batch_size=128,
            subset="validation")
        
        model.fit(train_generator, epochs=3)

        ypred = np.array(model.predict(test_generator))[:,:,0]
        
        ypreddf = pd.DataFrame(ypred.T, index=test_generator.index_array, columns=["x", "y", "z"])
        dfY = pd.read_csv("../data/train_labels.csv")[:count * 100]
        dfY_merged = pd.merge(ypreddf.drop(["x", "y", "z"], axis=1), dfY, left_index=True, right_index=True)
        dfYidx = dfY.drop(["x", "y", "z", "qw", "qx", "qy", "qz"], axis=1)
        df_merged = pd.merge(ypreddf, dfYidx, left_index=True, right_index=True)
        df_merged["qw"] = 0
        df_merged["qx"] = 0
        df_merged["qy"] = 0
        df_merged["qz"] = 0
        df_merged.loc[df_merged["i"] == 0, ["x", "y", "z", "qw", "qx", "qy", "qz"]] = [0,0,0,1,0,0,0]
        df_merged.set_index(["chain_id", "i"], inplace=True)
       # df_merged.drop(["id"], axis=1, inplace=True)
        ToCSV(df_merged, "cnn-test")
        ToCSV(dfY_merged, "cnn-true")

    return model

def make_submission_prediction():
    df_sub = pd.read_csv("../data/submission_format.csv")
    sub_datagen = ImageDataGenerator()
    
    # Create generator to read images from directories
    sub_generator = sub_datagen.flow_from_dataframe(dataframe=df_sub,
        directory="../data/images",
        x_col="path",
        y_col=["x", "y", "z", "qw", "qx", "qy", "qz"],
        target_size=(640, 512), # Downscale
        color_mode='grayscale',
        class_mode="raw",
        batch_size=64)
    
    sub_pred = model.predict(sub_generator, epochs=5)



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
conv = Conv2D(8, 8, activation='relu', kernel_initializer="he_normal", padding="same", input_shape=[640, 512, 1])(input)
norm = BatchNormalization()(conv)
pooling = MaxPooling2D(2)(conv)
conv2 = Conv2D(7, 5, activation='relu', kernel_initializer="he_normal", padding='same')(pooling)
flat = Flatten()(pooling)
hidden = Dense(128, activation="relu", kernel_initializer="he_normal")(flat)
#hidden = Dense(64, activation="relu")(hidden)
# Multiple outputs: https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras
x = Dense(1)(hidden)
y = Dense(1)(hidden)
z = Dense(1)(hidden)
# qw = Dense(1)(hidden)
# qx = Dense(1)(hidden)
# qy = Dense(1)(hidden)
# qz = Dense(1)(hidden)

model = Model(inputs=input, outputs=[x,y,z])

model.summary()
model.compile(loss="MSE", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.02))
load_images_and_train_model(30, model) # loading 50 chains trains on 5000 samples, takes ~130s per epoch (on GTX 1080ti)

# TODO: add testing for model