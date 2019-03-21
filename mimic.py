# Multiple Inputs
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.pooling import MaxPooling3D, MaxPooling2D
from keras import optimizers
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate, Input, TimeDistributed, LSTM, GRU, CuDNNGRU, \
    Recurrent, CuDNNLSTM, BatchNormalization, AlphaDropout, ConvLSTM2D, Activation, GaussianNoise, merge, Bidirectional
from keras import backend as K
from data_gen import *
import keras
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
import os
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
    # m_input = Input( batch_shape=(bsize, msize, msize, 7))
    m_input = Input(shape=(msize, msize, 7))
    # m_input_t = K.permute_dimensions(m_input, (0, 2, 3, 1))
    m_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='relu')(m_input)
    # maxpool
    m_conv2d1 = MaxPooling2D(pool_size=(2, 2))(m_conv2d1)
    m_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='relu')(m_conv2d1)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input = Input(shape=(ssize, ssize, 17))
    s_input_t = K.transpose(s_input)
    s_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='relu')(s_input)
    # maxpool
    s_conv2d1 = MaxPooling2D(pool_size=(2, 2))(s_conv2d1)
    s_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='relu')(s_conv2d1)

    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input = Input(shape=(infosize, 1))
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

    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=0.000001), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')

    return model


def build_model_keras_debug(infosize, msize, ssize, d):
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
    # m_input = Input( batch_shape=(bsize, msize, msize, 7))
    m_input1 = Input(shape=(1, msize, msize, 7))
    m_input = Reshape((msize, ssize, 7))(m_input1)
    # m_input_t = K.permute_dimensions(m_input, (0, 2, 3, 1))
    m_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='tanh')(m_input)
    # maxpool
    m_conv2d1 = MaxPooling2D(pool_size=(2, 2))(m_conv2d1)
    m_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='tanh')(m_conv2d1)
    m_conv2d2 = Dropout(d)(m_conv2d2)
    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input1 = Input(shape=(1, ssize, ssize, 17))
    s_input = Reshape((msize, ssize, 17))(s_input1)
    # s_input_t = K.transpose(s_input)
    s_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='tanh')(s_input)
    # maxpool
    s_conv2d1 = MaxPooling2D(pool_size=(2, 2))(s_conv2d1)
    s_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='tanh')(s_conv2d1)
    s_conv2d2 = Dropout(d)(s_conv2d2)
    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input = Input(shape=(1, infosize))
    i_input_f = Flatten()(i_input)
    i_fc = Dense(256, input_shape=(infosize,), activation='tanh')(i_input_f)
    i_fc = Dropout(d)(i_fc)
    # merge screen inputs

    # merge1 = Concatenate(axis=3)([m_conv2d2, s_conv2d2])
    merge2 = Concatenate(axis=1)([Flatten()(m_conv2d2), Flatten()(s_conv2d2), i_fc])

    # interpretation model
    hidden1 = Dense(256, activation='tanh')(merge2)
    value = Dense(1, activation='sigmoid')(hidden1)
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input1, s_input1, i_input], outputs=output)

    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model


# LSTM:
# Featurelayer eg conv
# MemoryLayer eg GRU
# Dense layer
def build_lstm_model(obs, infosize, msize, ssize, u, d):
    m_input = Input(shape=(obs, msize, msize, 7), name='minimap')

    # m_input_t = K.permute_dimensions(m_input, (0, 2, 3, 1))
    m_conv2d1 = Conv3D(16, kernel_size=5, strides=1, activation='tanh', name='minimap_conv2d1')(m_input)
    # m_conv2d1 = BatchNormalization()(m_conv2d1)
    # maxpool
    m_conv2d1 = MaxPooling3D(pool_size=(2, 2, 2))(m_conv2d1)
    m_conv2d2 = Conv3D(32, kernel_size=3, strides=1, activation='tanh', name='minimap_conv2d2')(m_conv2d1)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input = Input(shape=(obs, ssize, ssize, 17), name='screen')

    s_conv2d1 = Conv3D(16, kernel_size=5, strides=1, activation='tanh', name='screen_conv2d1')(s_input)
    # maxpool
    # s_conv2d1 = BatchNormalization()(s_conv2d1)
    s_conv2d1 = MaxPooling3D(pool_size=(2, 2, 2))(s_conv2d1)
    s_conv2d2 = Conv3D(32, kernel_size=3, strides=1, activation='tanh', name='screen_conv2d2')(s_conv2d1)

    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input = Input(shape=(obs, infosize), name='non-spatial')
    # i_input_f = Flatten()(i_input)

    i_fc = Dense(256, input_shape=(obs, infosize, 1), activation='tanh', name='non_spatial_dense')(i_input)

    model = Model(inputs=[m_input, s_input, i_input], outputs=s_conv2d2)
    print(model.output_shape)
    # merge screen inputs

    merge1 = Concatenate(axis=3)([m_conv2d2, s_conv2d2])
    # Reshape((32, 20, 28, 58))
    merge2 = Concatenate()([Reshape((32, 12 * 28 * 28))(m_conv2d2), Reshape((32, 12 * 28 * 28))(s_conv2d2), i_fc])

    # merge3 = Reshape((obs, int(1405184/obs)))(merge2)

    # interpretation model
    # merge3 = BatchNormalization()(merge3)
    hidden1 = LSTM(u, return_sequences=False, dropout=d)(merge2)
    value = Dense(1, activation='sigmoid')(hidden1)
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input, s_input, i_input], outputs=output)
    print(model.input_shape)
    print(model.output_shape)
    print(model.summary())
    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model


# sigmoid
# relu
# lstm
# relu
# lstm
# 53%
def build_lstm_model_debug2(obs, infosize, msize, ssize, u, d, lr):
    m_input = Input(shape=(obs, msize, msize, 7))
    m_conv2d2 = BatchNormalization()(m_input)
    m_conv2d2 = Dropout(d)(m_conv2d2)
    # filtering screen
    s_input = Input(shape=(obs, ssize, ssize, 17))
    s_conv2d2 = BatchNormalization()(s_input)
    s_conv2d2 = Dropout(d)(s_conv2d2)

    # merge1 = Concatenate(axis=-1)([m_conv2d2, s_conv2d2])

    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input1 = Input(shape=(obs, infosize), name='non-spatial')
    # i_input_f = Flatten()(i_input)
    i_input = BatchNormalization()(i_input1)
    i_fc = Dense(256, input_shape=(obs, infosize, 1), activation='tanh', name='non_spatial_dense')(i_input)
    i_fc = Dropout(d)(i_fc)
    # (None, 64, 256)

    # model = Model(inputs=[i_input1], outputs=i_fc)
    # print(model.output_shape)
    # merge screen inputs
    merge2 = Concatenate(axis=-1)([Reshape((obs, msize*msize*7))(m_conv2d2), Reshape((obs, ssize*ssize*17))(s_conv2d2), i_fc])

    # merge3 = Reshape((obs, int(1405184/obs)))(merge2)
    # model = Model(inputs=[m_input, s_input, i_input1], outputs=merge1)
    # print(model.input_shape)
    # print(model.output_shape)
    # interpretation model
    # merge3 = BatchNormalization()(merge3)
    # hidden1 = ConvLSTM2D(filters=u, kernel_size=5, return_sequences=False, dropout=d)(merge1)
    # merge2 = GaussianNoise(0.1)(merge2)
    hidden2 = Bidirectional(GRU(u, return_sequences=False, dropout=d, bias_initializer=keras.initializers.ones()))(merge2)
    # hidden2 = Bidirectional(LSTM(u, return_sequences=False, dropout=d/2))(hidden2)
    # merge3 = Concatenate()([hidden1, hidden2])

    # hidden1 = Flatten()(hidden1)
    value = Dense(1, activation='sigmoid', name="final_dense")(hidden2)
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input, s_input, i_input1], outputs=output)
    print(model.input_shape)
    print(model.output_shape)
    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model


def build_lstm_model_debug(obs, infosize, msize, ssize, u, d, lr, opt):
    # print(obs, infosize, msize, ssize)
    m_input1 = Input(shape=(obs, msize, msize, 7), name='minimap')
    m_conv2d1 = Conv3D(32, kernel_size=(1, 5, 5), strides=1, padding='same', activation='tanh',
                       bias_initializer=keras.initializers.ones())(m_input1)
    # m_input = Conv3D(32)
    m_input = GaussianNoise(0.1)(m_conv2d1)
    m_input = MaxPooling3D((1, 2, 2))(m_input)
    m_input = Dropout(d)(m_input)
    m_input = Conv3D(32, kernel_size=(1, 3, 3), strides=1, padding='same', activation='tanh',
                     bias_initializer=keras.initializers.ones())(m_input)
    m_input = Dropout(d)(m_input)
    # m_input = Dense(32, activation='tanh')(m_input)
    m_input = GaussianNoise(0.1)(m_input)
    m_input = BatchNormalization()(m_input)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input1 = Input(shape=(obs, ssize, ssize, 17), name='screen')
    s_conv2d1 = Conv3D(32, kernel_size=(1, 5, 5), strides=1, padding='same', activation='tanh',
                       bias_initializer=keras.initializers.ones())(s_input1)
    # m_input = Conv3D(32)
    s_input = GaussianNoise(0.1)(s_conv2d1)
    s_input = MaxPooling3D((1, 2, 2))(s_input)
    s_input = Dropout(d)(s_input)
    s_input = Conv3D(32, kernel_size=(1, 3, 3), strides=1, padding='same', activation='tanh',
                     bias_initializer=keras.initializers.ones())(s_input)
    s_input = Dropout(d)(s_input)
    # s_input = Dense(32, activation='tanh')(s_input)
    s_input = GaussianNoise(0.1)(s_input)
    s_input = BatchNormalization()(s_input)
    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input1 = Input(shape=(obs, infosize), name='non-spatial')
    # i_input_f = Flatten()(i_input)
    i_input = BatchNormalization()(i_input1)
    i_fc = Dense(256, input_shape=(obs, infosize, 1),
                 activation='tanh',
                 bias_initializer=keras.initializers.ones(),
                 # kernel_regularizer=regularizers.l1_l2(0.00001, 0.00001),
                 # activity_regularizer=regularizers.l1(0.00001),
                 name='non_spatial_dense'
                 )(i_input)
    i_fc = Dropout(d)(i_fc)
    # (None, 64, 256)
    # i_fc = Reshape

    # merge screen inputs
    # merge1 = Concatenate(axis=-1)([m_input, s_input])
    merge1 = Concatenate(axis=-1)([m_input1, s_input1])
    second_model = GaussianNoise(0.1)(merge1)
    second_model = Dense(64, activation='tanh', bias_initializer=keras.initializers.ones())(second_model)
    second_model = MaxPooling3D((1, 4, 4))(second_model)
    second_model = Dropout(d)(second_model)
    second_model = Reshape((obs, 16*16*64))(second_model)
    # model = Model(inputs=[m_input1, s_input1], outputs=second_model)
    # print(model.output_shape)
    hidden_GRU = GRU(u, dropout=d, return_sequences=False, bias_initializer=keras.initializers.ones())(second_model)

    # hidden_GRU = GRU(u, dropout=d, return_sequences=False, bias_initializer=keras.initializers.ones())(second_model)
    merge2 = Concatenate(axis=-1)([Reshape((obs, 32 * 32 * 32))(m_input), Reshape((obs, 32 * 32 * 32))(s_input), i_fc])

    # test_layer1 = Dense(512, activation='tanh', name='test_dense')(merge2)
    # merge3 = Reshape((obs, int(1405184/obs)))(merge2)

    # interpretation model
    # merge3 = BatchNormalization()(merge3)
    # hidden1 = LSTM(256, return_sequences=True, dropout=d)(test_layer1)

    # test_layer2 = Dense(128, activation='tanh', name='test_dense2')(hidden1)
    # hidden1 = ConvLSTM2D(16, kernel_size=5, strides=1, padding='same', name='hidden1', dropout=d,
    #                      return_sequences=True)(merge1)
    # hidden1 = BatchNormalization()(hidden1)
    # hidden2 = ConvLSTM2D(32, kernel_size=3, strides=1, padding='same', name='hidden2', dropout=d,
    #                      return_sequences=True)(hidden1)
    # hidden2 = BatchNormalization()(hidden2)
    # output1 = Conv3D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(hidden2)
    # hidden_LSTM2 = LSTM(u, dropout=d/2, return_sequences=False, bias_initializer=keras.initializers.ones())(merge2)
    # hidden_GRU = GRU(u, dropout=d, return_sequences=False, bias_initializer=keras.initializers.ones())(merge2)
    hidden_LSTM = LSTM(u, dropout=d, return_sequences=False, bias_initializer=keras.initializers.ones())(merge2)

    # ensemble = GaussianNoise(0.05)(ensemble)

    # hidden_nons = RNN(keras.layers.SimpleRNNCell(u, bias_initializer=keras.initializers.ones()), return_sequences=False)(merge2)
    # hidden_nons = Dense(u//2, activation='tanh', kernel_regularizer=regularizers.l2(1e-6))(hidden_nons)
    # hidden_nons = BatchNormalization()(hidden_nons)
    # hidden_nons = LSTM(u//4, return_sequences=False, dropout=d)(hidden_nons)
    # hidden_nons = Activation('tanh')(hidden_nons)
    # # 61 percent accuracy
    # hidden_nons = Dense(u, activation='relu',
    #                     # bias_initializer=keras.initializers.ones(),
    #
    #                     # kernel_regularizer=regularizers.l2(0.01)
    #                     # activity_regularizer=regularizers.l1(0.001),
    #                     )(hidden_nons)
    # hidden_nons = LSTM(u, return_sequences=True)(hidden_nons)
    # print(obs, u)
    # hidden_nons = Reshape((obs, u, 1))(hidden_nons)
    # hidden_nons = Conv2D(u//2, kernel_size=3, activation='relu',
    #                     # bias_initializer=keras.initializers.ones(),
    #                     # kernel_regularizer=regularizers.l2(0.01)
    #                     # activity_regularizer=regularizers.l1(0.001),
    #                     )(hidden_nons)
    # merge3 = Concatenate()([Flatten()(output1), hidden_nons])
    # model = Model(inputs=[m_input1, s_input1, i_input1], outputs=hidden1)
    # print(model.input_shape)
    # print(model.output_shape)
    # hidden_nons = Flatten()(hidden_nons)

    value1 = Dense(1, activation='sigmoid', name="final_dense1", bias_initializer=keras.initializers.ones())(hidden_GRU)
    value2 = Dense(1, activation='sigmoid', name="final_dense2", bias_initializer=keras.initializers.ones())(hidden_LSTM)
    ensemble = merge.Average()([value1, value2])
    value = Reshape((1,))(ensemble)
    model = Model(inputs=[m_input1, s_input1, i_input1], outputs=value)
    print(model.summary())
    print(model.input_shape)
    print(model.output_shape)
    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model


def build_lstm_model_freeze1(obs, infosize, msize, ssize, u, d, lr):
    # print(obs, infosize, msize, ssize)
    m_input1 = Input(shape=(obs, msize, msize, 7), name='minimap')
    m_conv2d1 = Conv3D(16, kernel_size=(1, 2, 2), strides=1, padding='same', activation='tanh',
                       bias_initializer=keras.initializers.ones())(m_input1)
    # m_input = Conv3D(32)
    m_input = GaussianNoise(0.05)(m_conv2d1)
    m_input = MaxPooling3D((1, 2, 2))(m_input)
    m_input = Dense(32, activation='tanh')(m_input)
    m_input = GaussianNoise(0.05)(m_input)
    m_input = BatchNormalization()(m_input)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input1 = Input(shape=(obs, ssize, ssize, 17), name='screen')
    s_conv2d1 = Conv3D(16, kernel_size=(1, 2, 2), strides=1, padding='same', activation='tanh',
                       bias_initializer=keras.initializers.ones())(s_input1)
    # m_input = Conv3D(32)
    s_input = GaussianNoise(0.05)(s_conv2d1)
    s_input = MaxPooling3D((1, 2, 2))(s_input)
    s_input = Dense(32, activation='tanh')(s_input)
    s_input = GaussianNoise(0.05)(s_input)
    s_input = BatchNormalization()(s_input)
    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input1 = Input(shape=(obs, infosize), name='non-spatial')
    # i_input_f = Flatten()(i_input)
    i_input = BatchNormalization()(i_input1)
    i_fc = Dense(256, input_shape=(obs, infosize, 1),
                 activation='tanh',
                 bias_initializer=keras.initializers.ones(),
                 # kernel_regularizer=regularizers.l1_l2(0.00001, 0.00001),
                 # activity_regularizer=regularizers.l1(0.00001),
                 name='non_spatial_dense'
                 )(i_input)
    # (None, 64, 256)

    model = Model(inputs=[m_input1], outputs=m_input)
    print(model.output_shape)
    # merge screen inputs
    # merge1 = Concatenate(axis=-1)([m_input, s_input])
    merge2 = Concatenate(axis=-1)([Reshape((obs, 32 * 32 * 32))(m_input), Reshape((obs, 32 * 32 * 32))(s_input), i_fc])

    # test_layer1 = Dense(512, activation='tanh', name='test_dense')(merge2)
    # merge3 = Reshape((obs, int(1405184/obs)))(merge2)

    # interpretation model
    # merge3 = BatchNormalization()(merge3)
    # hidden1 = LSTM(256, return_sequences=True, dropout=d)(test_layer1)

    # test_layer2 = Dense(128, activation='tanh', name='test_dense2')(hidden1)
    # hidden1 = ConvLSTM2D(16, kernel_size=5, strides=1, padding='same', name='hidden1', dropout=d,
    #                      return_sequences=True)(merge1)
    # hidden1 = BatchNormalization()(hidden1)
    # hidden2 = ConvLSTM2D(32, kernel_size=3, strides=1, padding='same', name='hidden2', dropout=d,
    #                      return_sequences=True)(hidden1)
    # hidden2 = BatchNormalization()(hidden2)
    # output1 = Conv3D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(hidden2)
    # hidden_nons = GRU(u, return_sequences=False, bias_initializer=keras.initializers.ones(),dropout=d)(merge2)
    hidden_nons = Bidirectional(GRU(u, return_sequences=True, dropout=d, bias_initializer=keras.initializers.ones()))(merge2)
    # hidden_nons = Dense(u//2, activation='tanh', kernel_regularizer=regularizers.l2(1e-6))(hidden_nons)
    # hidden_nons = BatchNormalization()(hidden_nons)
    # hidden_nons = LSTM(u//4, return_sequences=False, dropout=d)(hidden_nons)
    # hidden_nons = Activation('tanh')(hidden_nons)
    # # 61 percent accuracy
    # hidden_nons = Dense(u, activation='relu',
    #                     # bias_initializer=keras.initializers.ones(),
    #
    #                     # kernel_regularizer=regularizers.l2(0.01)
    #                     # activity_regularizer=regularizers.l1(0.001),
    #                     )(hidden_nons)
    # hidden_nons = LSTM(u, return_sequences=True)(hidden_nons)
    # print(obs, u)
    # hidden_nons = Reshape((obs, u, 1))(hidden_nons)
    # hidden_nons = Conv2D(u//2, kernel_size=3, activation='relu',
    #                     # bias_initializer=keras.initializers.ones(),
    #                     # kernel_regularizer=regularizers.l2(0.01)
    #                     # activity_regularizer=regularizers.l1(0.001),
    #                     )(hidden_nons)
    # merge3 = Concatenate()([Flatten()(output1), hidden_nons])
    # model = Model(inputs=[m_input1, s_input1, i_input1], outputs=hidden1)
    # print(model.input_shape)
    # print(model.output_shape)
    # hidden_nons = Flatten()(hidden_nons)
    value = Dense(1, activation='sigmoid', name="final_dense", bias_initializer=keras.initializers.ones())(hidden_nons)
    value = Reshape((1,))(value)
    model = Model(inputs=[m_input1, s_input1, i_input1], outputs=value)
    print(model.summary())
    print(model.input_shape)
    print(model.output_shape)
    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model

# 57
def build_lstm_model_freeze2(obs, infosize, msize, ssize, u, d, lr):
    # print(obs, infosize, msize, ssize)
    m_input1 = Input(shape=(obs, msize, msize, 7), name='minimap')
    m_conv2d1 = Conv3D(16, kernel_size=(1, 2, 2), strides=1, padding='same', activation='tanh',
                       bias_initializer=keras.initializers.ones())(m_input1)
    # m_input = Conv3D(32)
    m_input = GaussianNoise(0.05)(m_conv2d1)
    m_input = MaxPooling3D((1, 2, 2))(m_input)
    m_input = Dense(32, activation='tanh')(m_input)
    m_input = GaussianNoise(0.1)(m_input)
    m_input = BatchNormalization()(m_input)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input1 = Input(shape=(obs, ssize, ssize, 17), name='screen')
    s_conv2d1 = Conv3D(16, kernel_size=(1, 2, 2), strides=1, padding='same', activation='tanh',
                       bias_initializer=keras.initializers.ones())(s_input1)
    # m_input = Conv3D(32)
    s_input = GaussianNoise(0.1)(s_conv2d1)
    s_input = MaxPooling3D((1, 2, 2))(s_input)
    s_input = Dense(32, activation='tanh')(s_input)
    s_input = GaussianNoise(0.05)(s_input)
    s_input = BatchNormalization()(s_input)
    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input1 = Input(shape=(obs, infosize), name='non-spatial')
    # i_input_f = Flatten()(i_input)
    i_input = BatchNormalization()(i_input1)
    i_fc = Dense(256, input_shape=(obs, infosize, 1),
                 activation='tanh',
                 bias_initializer=keras.initializers.ones(),
                 # kernel_regularizer=regularizers.l1_l2(0.00001, 0.00001),
                 # activity_regularizer=regularizers.l1(0.00001),
                 name='non_spatial_dense'
                 )(i_input)
    # (None, 64, 256)

    model = Model(inputs=[m_input1], outputs=m_input)
    print(model.output_shape)
    # merge screen inputs
    # merge1 = Concatenate(axis=-1)([m_input, s_input])
    merge2 = Concatenate(axis=-1)([Reshape((obs, 32 * 32 * 32))(m_input), Reshape((obs, 32 * 32 * 32))(s_input), i_fc])

    # test_layer1 = Dense(512, activation='tanh', name='test_dense')(merge2)
    # merge3 = Reshape((obs, int(1405184/obs)))(merge2)

    # interpretation model
    # merge3 = BatchNormalization()(merge3)
    # hidden1 = LSTM(256, return_sequences=True, dropout=d)(test_layer1)

    # test_layer2 = Dense(128, activation='tanh', name='test_dense2')(hidden1)
    # hidden1 = ConvLSTM2D(16, kernel_size=5, strides=1, padding='same', name='hidden1', dropout=d,
    #                      return_sequences=True)(merge1)
    # hidden1 = BatchNormalization()(hidden1)
    # hidden2 = ConvLSTM2D(32, kernel_size=3, strides=1, padding='same', name='hidden2', dropout=d,
    #                      return_sequences=True)(hidden1)
    # hidden2 = BatchNormalization()(hidden2)
    # output1 = Conv3D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(hidden2)
    # hidden_nons = GRU(u, return_sequences=False, bias_initializer=keras.initializers.ones(),dropout=d)(merge2)
    hidden_nons = Bidirectional(GRU(u, return_sequences=True, dropout=d, bias_initializer=keras.initializers.ones()))(merge2)
    hidden_nons = Bidirectional(LSTM(u, return_sequences=False, bias_initializer=keras.initializers.ones()))(hidden_nons)
    # hidden_nons = Dense(u//2, activation='tanh', kernel_regularizer=regularizers.l2(1e-6))(hidden_nons)
    # hidden_nons = BatchNormalization()(hidden_nons)
    # hidden_nons = LSTM(u//4, return_sequences=False, dropout=d)(hidden_nons)
    # hidden_nons = Activation('tanh')(hidden_nons)
    # # 61 percent accuracy
    # hidden_nons = Dense(u, activation='relu',
    #                     # bias_initializer=keras.initializers.ones(),
    #
    #                     # kernel_regularizer=regularizers.l2(0.01)
    #                     # activity_regularizer=regularizers.l1(0.001),
    #                     )(hidden_nons)
    # hidden_nons = LSTM(u, return_sequences=True)(hidden_nons)
    # print(obs, u)
    # hidden_nons = Reshape((obs, u, 1))(hidden_nons)
    # hidden_nons = Conv2D(u//2, kernel_size=3, activation='relu',
    #                     # bias_initializer=keras.initializers.ones(),
    #                     # kernel_regularizer=regularizers.l2(0.01)
    #                     # activity_regularizer=regularizers.l1(0.001),
    #                     )(hidden_nons)
    # merge3 = Concatenate()([Flatten()(output1), hidden_nons])
    # model = Model(inputs=[m_input1, s_input1, i_input1], outputs=hidden1)
    # print(model.input_shape)
    # print(model.output_shape)
    # hidden_nons = Flatten()(hidden_nons)
    value = Dense(1, activation='sigmoid', name="final_dense", bias_initializer=keras.initializers.ones())(hidden_nons)
    value = Reshape((1,))(value)
    model = Model(inputs=[m_input1, s_input1, i_input1], outputs=value)
    print(model.summary())
    print(model.input_shape)
    print(model.output_shape)
    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model
# def ensemble(models: List[training.Model], model_input: Tensor) -> training.Model:
#     outputs = [model.outputs[0] for model in models]
#     y = Average()(outputs)
#
#     model = Model(model_input, y, name='ensemble')
#
#     return model

# best, obs=32, batch_size=10, u=128, dropout=0.2
# best, obs=32, batch_size=7, u=64, dropout=0.2, lr = 1e-4, 59%
if __name__ == '__main__':
    obs = [16]
    batch_size = [40]
    units = [128]
    dropout = [0.2]
    lr = [1e-4]
    input_data = './replay_data/'
    eval_test = './pick_data/'
    small_eval = './extra/'
    even_smaller = './small/'
    opts = [optimizers.SGD(lr=lr[0])]
    # opts = [keras.optimizers.Adagrad(lr=lr[0]), keras.optimizers.RMSprop(lr=lr[0]), keras.optimizers.Nadam(lr=lr[0])]
    accuracies = []
    # if os.path.isfile('primitive_train_SGD.h5'):

    # model = load_model('lstm3d_8_48_128_0.2_with_0.7291666666666666.h5')
    # eval_generator = Mygenerator(batch_size[0], obs[0], input_data)
    # print('length of generator = ', len(eval_generator))
    # score = model.evaluate_generator(eval_generator, workers=1)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    #
    #     print(model.predict_generator(eval_generator))
    #     weights, biases = model.layers[0].get_weights()
    #     print(weights,biases)
    #     model = load_model('primitive_train2.h5')
    #     print("printing adam")
    #     print(model.predict_generator(eval_generator))
    #     weights, biases = model.layers[0].get_weights()
    #     print(weights, biases)
    # test_replays = './test_replays/Replays/'
    # else:
    # model = build_lstm_model(552, 64, 64)
    # model = build_lstm_model_debug(obs, 552, 64, 64)
    # model2 = build_lstm_model_debug(32, 552, 64, 64)
    # model3 = build_lstm_model_debug(16, 552, 64, 64)



    # PATH_REPLAY = 'D:/University_Work/My_research/fixed_replays/Replays/'
    # model = load_model('primitive_train_tanh.h5')
    # batch_size, obs, replay_path
    for b in batch_size:
        for o in obs:
            for u in units:
                for d in dropout:
                    for opt in opts:
                        # model = build_model_keras_debug(552, 64, 64, d)
                        # model = build_lstm_model_freeze2(o, 552, 64, 64, u, d, lr[0])
                        # model = load_model('actual_lstm3d_b64_next0.5382353007501247.h5')
                        model = load_model('LSTM3D16_next57.h5')

                        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=b,
                                                                  write_graph=True, write_grads=False,
                                                                  write_images=False, embeddings_freq=0,
                                                                  embeddings_layer_names=None,
                                                                  embeddings_metadata=None, embeddings_data=None)
                        # training_generator = Mygenerator(b, o, eval_test)
                        # eval_generator = Mygenerator(b, o, small_eval)

                        training_generator = Mygenerator(b, o, input_data)
                        eval_generator = Mygenerator(b, o, eval_test)

                        # training_generator = Mygenerator(b, o, small_eval)
                        # eval_generator = Mygenerator(b, o, even_smaller)

                        predict_generator = Mygenerator(b, o, even_smaller)
                        # model.fit_generator(generator=training_generator,
                        #                     validation_data=eval_generator,
                        #                     workers=1,
                        #                     epochs=10,
                        #                     # steps_per_epoch=20,
                        #                     # validation_steps=5,
                        #                     # max_queue_size=20,
                        #                     shuffle=True,
                        #                     # steps_per_epoch=2,
                        #                     use_multiprocessing=False,
                        #                     callbacks=[tensorboard]
                        #                     )
                        # model2.fit_generator(generator=training_generator,
                        #                     validation_data=eval_generator,
                        #                     workers=6,
                        #                     epochs=5,
                        #                     shuffle=True,
                        #                     # steps_per_epoch=2,
                        #                     use_multiprocessing=True,
                        #                     callbacks=[tensorboard]
                        #                     )
                        # predictions = model.predict_generator(predict_generator)
                        score = model.evaluate_generator(predict_generator, workers=1)
                        accuracies.append((score[0], score[1]))
                        print('pick_data loss:', score[0])
                        print('pick_data accuracy:', score[1])
                        #
                        # predict_generator = Mygenerator(b, o, input_data)
                        # predictions = model.predict_generator(predict_generator)
                        # score = model.evaluate_generator(eval_generator, workers=1)
                        # print('replay_data loss:', score[0])
                        # print('replay_data accuracy:', score[1])
                        # # print(predictions)
                        #
                        model.save(
                            'LSTM3D{1}_next{3}.h5'.format(str(b), str(o), str(u), str(score[1]), str(d)))
                        # model.save('actual_lstm3d_b{1}_next{3}.h5'.format(str(b), str(o), str(u), str(score[1]), str(d)))
                        # model.save('primitive_train_lstm2.h5')

    # for b in batch_size:
    #     for o in obs:
    #         for u in units:
    #             for d in dropout:
    #                 for opt in opts:
    #                     model = build_lstm_model_debug2(o, 552, 64, 64, u, d, lr[0])
    #                     tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=b,
    #                                                               write_graph=True, write_grads=False,
    #                                                               write_images=False, embeddings_freq=0,
    #                                                               embeddings_layer_names=None,
    #                                                               embeddings_metadata=None, embeddings_data=None)
    #                     # training_generator = Mygenerator(b, o, eval_test)
    #                     # eval_generator = Mygenerator(b, o, small_eval)
    #
    #                     training_generator = Mygenerator(b, o, input_data)
    #                     eval_generator = Mygenerator(b, o, eval_test)
    #
    #                     # training_generator = Mygenerator(b, o, input_data)
    #                     # eval_generator = Mygenerator(b, o, eval_test)
    #
    #                     predict_generator = Mygenerator(b, o, small_eval)
    #                     model.fit_generator(generator=training_generator,
    #                                         validation_data=eval_generator,
    #                                         workers=1,
    #                                         epochs=10,
    #                                         shuffle=True,
    #                                         steps_per_epoch=20,
    #                                         validation_steps=5,
    #                                         use_multiprocessing=False,
    #                                         callbacks=[tensorboard]
    #                                         )
    #                     # model2.fit_generator(generator=training_generator,
    #                     #                     validation_data=eval_generator,
    #                     #                     workers=6,
    #                     #                     epochs=5,
    #                     #                     shuffle=True,
    #                     #                     # steps_per_epoch=2,
    #                     #                     use_multiprocessing=True,
    #                     #                     callbacks=[tensorboard]
    #                     #                     )
    #                     # predictions = model.predict_generator(predict_generator)
    #                     score = model.evaluate_generator(predict_generator, workers=1)
    #                     accuracies.append((score[0], score[1]))
    #                     print('pick_data loss:', score[0])
    #                     print('pick_data accuracy:', score[1])
                        #
                        # predict_generator = Mygenerator(b, o, input_data)
                        # predictions = model.predict_generator(predict_generator)
                        # score = model.evaluate_generator(eval_generator, workers=1)
                        # print('replay_data loss:', score[0])
                        # print('replay_data accuracy:', score[1])
                        # # print(predictions)
                        #
                        # model.save('lstm3d_{0}_{1}_{2}_{4}_with_{3}.h5'.format(str(b), str(o), str(u), str(score[1]), str(d)))
                        # model.save('primitive_train_lstm2.h5')
    print(accuracies)