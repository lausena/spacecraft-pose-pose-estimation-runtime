import pandas as pd

import tensorflow as tf
from keras.models import Sequential

import keras
import keras.preprocessing.image as kimg

from keras import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate, Flatten, Conv2D, MaxPooling2D, AvgPool2D

from PIL import Image

from models.data_generator import CustomDataGenerator


def train_translational_model(dataframe_dict):

	feature_dataframe = dataframe_dict["features"]
	translation_dataframe = dataframe_dict["translation"]

	# Drop Pose Columns, Set y label
	# pos_labels = pd.DataFrame(pos_labels, columns=['x', 'y', 'z'])

	# Build Model
	model_pos = build_CNN_Model()

	# Create Custom Data Generator
	data_generator = CustomDataGenerator(feature_dataframe, translation_dataframe, batch_size=100)

	X, y = data_generator[0]

	print(X)
	print(y)

	# Train Model
	with tf.device('/GPU:0'):
		model_pos.fit(data_generator, epochs=3)

def build_CNN_Model():

	# Input Shapes
	conv_input_shape = (512, 640, 3)
	range_input_shape = (1, 1)

	# Define Model

	## Inputs
	conv_branch_input_Base = Input(shape=conv_input_shape)
	conv_branch_input_Img = Input(shape=conv_input_shape)
	# range_input = Input(shape=range_input_shape)                                        ### Uncomment and add to concat when working

	## Convolutional Branches
	conv_branch = create_conv_branch_pos(conv_input_shape)
	processed_conv_A = conv_branch(conv_branch_input_Base)
	processed_conv_B = conv_branch(conv_branch_input_Img)

	## Dense Layers
	conv_concat = keras.layers.concatenate([processed_conv_A, processed_conv_B])
	conv_flatten_layer = Flatten()(conv_concat)

	# concatenated_layer = keras.layers.concatenate([conv_flatten_layer, range_input])   ### Add Range Here when working

	dropout_layer1 = Dropout(0.25)(conv_flatten_layer)
	dense_layer1 = Dense(16)(dropout_layer1)

	dropout_layer2 = Dropout(0.25)(dense_layer1)
	dense_layer2 = Dense(16)(dropout_layer2)
	
	dropout_layer3 = Dropout(0.25)(dense_layer2)
	x1 = Dense(16)(dropout_layer3)

	output = Dense(3)(x1)

	## Create Model
	model_pos = Model(inputs=[conv_branch_input_Base, conv_branch_input_Img], outputs=output)
	model_pos.compile(optimizer=Adam(learning_rate=0.0001), loss="log_cosh")
	model_pos.summary()

	return model_pos

# from https://github.com/1988kramer/camera-pose/blob/master/camera-pose.py 
def create_conv_branch(input_shape):
	model = Sequential()
	model.add(Conv2D(96, kernel_size=(11,11),
					 strides=4, padding='valid',
					 activation='relu',
					 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Conv2D(128, kernel_size=(5,5),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=1))
	model.add(Conv2D(256, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(256, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(128, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	return model

def create_conv_branch_pos(input_shape):
	model = Sequential()
	model.add(Conv2D(96, kernel_size=(11,11),
					 strides=4, padding='valid',
					 activation='relu',
					 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Conv2D(128, kernel_size=(7,7),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=1))
	model.add(Conv2D(256, kernel_size=(5,5),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(256, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(128, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=2))
	model.add(Conv2D(256, kernel_size=(5,5),
					 strides=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(128, kernel_size=(3,3),
					 strides=1, padding='same',
					 activation='relu'))

	return model