import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf, tensorflow.keras as keras
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.models import Model
import cv2
import os
import pandas as pd

from tensorflow.keras.utils import image_dataset_from_directory as idfd


dfY = pd.read_csv("../data/train_labels.csv")
data = []
dataY = []

def load_images(count):
    j = 0
    for dir in os.listdir("../data/images"):
        if j == count:
            break
        for img in os.listdir(f"../data/images/{dir}"):
            #print(img)
            data.append([tf.cast(cv2.imread(f"../data/images/{dir}/{img}", 0), dtype=tf.float32)])
            i = int(img[:3])
            #print(dfY.loc[(dfY.chain_id == dir) & (dfY.i == i), ["x", "y", "z", "qw", "qx", "qy", "qz"]].values)
            dataY.append(dfY.loc[(dfY.chain_id == dir) & (dfY.i == i), ["x", "y", "z", "qw", "qx", "qy", "qz"]].values.tolist()[0])
        j += 1
    #print(data)
    return np.array(data).reshape(-1, 1280, 1024, 1), pd.DataFrame(dataY, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])

dfX, dfY = load_images(10)
# x_train = dfX[:9]
# y_train = dfY[:9]
# x_test = dfX[-9:]
# y_test = dfY[-9:]
print(dfY)
print(dfX.shape)
dfY.fillna(value=0, inplace=True)
# plt.imshow(dfX.tolist(), cmap="gray") # plot 1st image's 2nd feature map
# plt.show()


input = Input(shape=(1280,1024,1))
conv = Conv2D(4, 4, activation='relu', padding="same", input_shape=[1280, 1024, 1])(input)
#BatchNormalization()(conv)
pooling = MaxPooling2D(2)(conv)
conv2 = Conv2D(3, 2, activation='relu', padding='same')(pooling)
flat = Flatten()(conv2)
hidden = Dense(16, activation="relu")(flat)
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
model.fit(dfX, [dfY.loc[:, "x"], dfY.loc[:, "y"], dfY.loc[:, "z"], dfY.loc[:, "qw"], dfY.loc[:, "qx"], dfY.loc[:, "qy"], dfY.loc[:, "qz"]], epochs=10)