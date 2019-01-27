# Multiple Inputs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate, Input, TimeDistributed, LSTM
from keras import backend as K
from data_gen import *

import tensorflow as tf


def build_model_keras(infosize, msize, ssize):
    # # filtering minimap
    # pre_model_m = Sequential()
    # K.transpose(minimap)
    # pre_model_m.add(Conv2D(16, kernel_size=5, strides=1, input_shape=(msize, msize, 7)))
    # pre_model_m.add(Conv2D(32, kernel_size=4, strides=2))
    # # filtering screen
    # pre_model_s = Sequential()
    # K.transpose(screen)
    # pre_model_s.add(Conv2D(16, kernel_size=8, strides=4, input_shape=(ssize, ssize, 17)))
    # pre_model_s.add(Conv2D(32, kernel_size=4, strides=2))
    #
    # # flatten and filter non-spatial features
    # pre_model_i = Sequential()
    # pre_model_i.add(Flatten())
    # pre_model_i.add(Dense(256, input_shape=(len(info),), activation='tanh'))
    #
    # concat_inputs = Concatenate()([pre_model_s, pre_model_s, info], axis=1)
    #
    # model = Sequential()
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(1))
    # model.add(Reshape(1))

    # filtering minimap

    # msize, ssize, infosize = 64, 64, 541
    # original
    # visible1 = Input(shape=(64,64,1))
    # conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
    # pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    # conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
    # pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    # flat1 = Flatten()(pool12)

    # filtering minimap
    m_input = Input( batch_shape=(64, msize, msize, 7))
    # m_input_t = K.permute_dimensions(m_input, (0, 2, 3, 1))
    m_conv2d1 = Conv2D(16, kernel_size=5, strides=1)(m_input)
    # maxpool
    m_conv2d1 = MaxPooling2D(pool_size=(2, 2))(m_conv2d1)
    m_conv2d2 = Conv2D(32, kernel_size=3, strides=1)(m_conv2d1)

    # filtering screen
    s_input = Input(batch_shape=(64, ssize, ssize, 17))
    s_input_t = K.transpose(s_input)
    s_conv2d1 = Conv2D(16, kernel_size=5, strides=1)(s_input)
    # maxpool
    s_conv2d1 = MaxPooling2D(pool_size=(2, 2))(s_conv2d1)
    s_conv2d2 = Conv2D(32, kernel_size=3, strides=1)(s_conv2d1)

    # info tensor
    i_input = Input( batch_shape=(64, infosize, 1))
    i_input_f = Flatten()(i_input)
    i_fc = Dense(256, input_shape=(infosize,), activation='tanh')(i_input_f)

    # merge screen inputs

    merge1 = Concatenate(axis=3)([m_conv2d2, s_conv2d2])
    merge2 = Concatenate(axis=1)([Flatten()(m_conv2d2), Flatten()(s_conv2d2), i_fc])

    # interpretation model
    hidden1 = Dense(256, activation='relu')(merge2)
    value = Dense(1, activation='sigmoid')(hidden1)
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input, s_input, i_input], outputs=output)

    model.compile(optimizer=optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# LSTM:
# Featurelayer eg conv
# MemoryLayer eg GRU
# Dense layer
def build_lstm_model(infosize, msize, ssize):
    # filtering minimap
    m_input = Input( batch_shape=(64, msize, msize, 7))
    # m_input_t = K.permute_dimensions(m_input, (0, 2, 3, 1))
    m_conv2d1 = Conv2D(16, kernel_size=5, strides=1)(m_input)
    # maxpool
    m_conv2d1 = MaxPooling2D(pool_size=(2, 2))(m_conv2d1)
    m_conv2d2 = Conv2D(32, kernel_size=3, strides=1)(m_conv2d1)

    # filtering screen
    s_input = Input(batch_shape=(64, ssize, ssize, 17))
    # s_input_t = K.transpose(s_input)
    s_conv2d1 = Conv2D(16, kernel_size=5, strides=1)(s_input)
    # maxpool
    s_conv2d1 = MaxPooling2D(pool_size=(2, 2))(s_conv2d1)
    s_conv2d2 = Conv2D(32, kernel_size=3, strides=1)(s_conv2d1)

    # info tensor
    i_input = Input( batch_shape=(64, infosize, 1))
    i_input_f = Flatten()(i_input)
    i_fc = Dense(256, input_shape=(infosize,), activation='tanh')(i_input_f)

    # merge screen inputs

    merge1 = Concatenate(axis=3)([m_conv2d2, s_conv2d2])
    merge2 = Concatenate(axis=1)([Flatten()(m_conv2d2), Flatten()(s_conv2d2), i_fc])

    lstm = TimeDistributed(Dense(256, activation='tanh')(merge2))
    lstm = LSTM(10, activation='sigmoid', )
    # interpretation model
    # hidden1 = Dense(256, activation='relu')(merge2)
    value = TimeDistributed(Dense(1, activation='sigmoid')(lstm))
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input, s_input, i_input], outputs=output)

    model.compile(optimizer=optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = build_model_keras(541+11, 64, 64)

batch_size = 1
test_replays = './test_replays/Replays/'
PATH_REPLAY = 'D:/University_Work/My_research/fixed_replays/Replays/'
training_generator = Mygenerator(batch_size, test_replays)

model.fit_generator(generator=training_generator, workers=4)
