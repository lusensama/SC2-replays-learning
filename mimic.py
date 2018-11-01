import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

def build_model_keras(minimap, screen, info, msize, ssize, num_action):

    # filtering minimap
    pre_model_m = Sequential()
    K.transpose(minimap)
    pre_model_m.add(Conv2D(16, kernel_size=5, strides=1, input_shape=(msize, msize, 7)))
    pre_model_m.add(Conv2D(32, kernel_size=4, strides=2))
    # filtering screen
    pre_model_s = Sequential()
    K.transpose(screen)
    pre_model_s.add(Conv2D(16, kernel_size=8, strides=4, input_shape=(ssize, ssize, 17)))
    pre_model_s.add(Conv2D(32, kernel_size=4, strides=2))

    # flatten and filter non-spatial features
    pre_model_i = Sequential()
    pre_model_i.add(Flatten())
    pre_model_i.add(Dense(256, input_shape=(len(info),), activation='tanh'))

    K.concatenate([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)

    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    model.add(Reshape(1))

