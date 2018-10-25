import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

def build_model_keras(minimap, screen, info, msize, ssize, num_action):
    mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=8,
                           stride=4,
                           scope='mconv1')
    mconv2 = layers.conv2d(mconv1,
                           num_outputs=32,
                           kernel_size=4,
                           stride=2,
                           scope='mconv2')
    sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=8,
                           stride=4,
                           scope='sconv1')
    sconv2 = layers.conv2d(sconv1,
                           num_outputs=32,
                           kernel_size=4,
                           stride=2,
                           scope='sconv2')
    info_fc = layers.fully_connected(layers.flatten(info),
                                     num_outputs=256,
                                     activation_fn=tf.tanh,
                                     scope='info_fc')
    feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                     num_outputs=256,
                                     activation_fn=tf.nn.relu,
                                     scope='feat_fc')
    value = tf.reshape(layers.fully_connected(feat_fc,
                                              num_outputs=1,
                                              activation_fn=None,
                                              scope='value'), [-1])



    model = Sequential()
    tf.transpose(minimap, [0, 2, 3, 1])
    model.add(Conv2D(16, kernel_size=8, strides=4,
                     input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=4, strides=2,
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2, activation='tanh'))