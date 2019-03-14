# Multiple Inputs
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate, Input, TimeDistributed, LSTM, GRU, CuDNNGRU, Recurrent, CuDNNLSTM
from keras import backend as K
from data_gen import *
import keras
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
    model.compile(optimizer=optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')

    return model


def build_model_keras_debug(infosize, msize, ssize):
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
    m_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='tanh')(m_input)
    # maxpool
    m_conv2d1 = MaxPooling2D(pool_size=(2, 2))(m_conv2d1)
    m_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='tanh')(m_conv2d1)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input = Input(shape=(ssize, ssize, 17))
    s_input_t = K.transpose(s_input)
    s_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='tanh')(s_input)
    # maxpool
    s_conv2d1 = MaxPooling2D(pool_size=(2, 2))(s_conv2d1)
    s_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='tanh')(s_conv2d1)

    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input = Input(shape=(infosize, 1))
    i_input_f = Flatten()(i_input)
    i_fc = Dense(256, input_shape=(infosize,), activation='tanh')(i_input_f)

    # merge screen inputs

    merge1 = Concatenate(axis=3)([m_conv2d2, s_conv2d2])
    merge2 = Concatenate(axis=1)([Flatten()(m_conv2d2), Flatten()(s_conv2d2), i_fc])

    # interpretation model
    hidden1 = Dense(256, activation='tanh')(merge2)
    value = Dense(1, activation='sigmoid')(hidden1)
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input, s_input, i_input], outputs=output)

    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model
# LSTM:
# Featurelayer eg conv
# MemoryLayer eg GRU
# Dense layer
def build_lstm_model(infosize, msize, ssize):
    m_input = Input(shape=(msize, msize, 7))

    # m_input_t = K.permute_dimensions(m_input, (0, 2, 3, 1))
    m_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='tanh')(m_input)
    # m_conv2d1 = BatchNormalization()(m_conv2d1)
    # maxpool
    m_conv2d1 = MaxPooling2D(pool_size=(2, 2))(m_conv2d1)
    m_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='tanh')(m_conv2d1)

    # filtering screen
    # s_input = Input(batch_shape=(bsize, ssize, ssize, 17))
    s_input = Input(shape=(ssize, ssize, 17))

    s_conv2d1 = Conv2D(16, kernel_size=5, strides=1, activation='tanh')(s_input)
    # maxpool
    # s_conv2d1 = BatchNormalization()(s_conv2d1)
    s_conv2d1 = MaxPooling2D(pool_size=(2, 2))(s_conv2d1)
    s_conv2d2 = Conv2D(32, kernel_size=3, strides=1, activation='tanh')(s_conv2d1)

    # info tensor
    # i_input = Input( batch_shape=(bsize, infosize, 1))
    i_input = Input(shape=(infosize, 1))
    i_input_f = Flatten()(i_input)
    i_fc = Dense(256, input_shape=(infosize,), activation='tanh')(i_input_f)

    # merge screen inputs

    merge1 = Concatenate(axis=3)([m_conv2d2, s_conv2d2])
    merge2 = Concatenate(axis=1)([Flatten()(m_conv2d2), Flatten()(s_conv2d2), i_fc])
    merge3 = Reshape((1, 50432))(merge2)
    # interpretation model
    # merge3 = BatchNormalization()(merge3)
    hidden1 = CuDNNLSTM(8, return_sequences=False, input_shape=(64, 50432))(merge3)
    value = Dense(1, activation='sigmoid')(hidden1)
    output = Reshape((1,))(value)
    model = Model(inputs=[m_input, s_input, i_input], outputs=output)
    print(model.input_shape)
    print(model.output_shape)
    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    return model

if __name__ == '__main__':
    batch_size = 64
    input_data = './replay_data/'
    eval_test = './pick_data/'
    # if os.path.isfile('primitive_train_SGD.h5'):
    #
    #     model = load_model('primitive_train_SGD.h5')
    #     eval_generator = Mygenerator(batch_size, eval_test)
    #     print('length of generator = ', len(eval_generator))
    #     score = model.evaluate_generator(eval_generator, workers=1)
    #     print('Test loss:', score[0])
    #     print('Test accuracy:', score[1])
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
    model = build_lstm_model(552, 64, 64)
    # model2 = build_lstm_model2(552, 64, 64)

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None)
    # PATH_REPLAY = 'D:/University_Work/My_research/fixed_replays/Replays/'
    # model = load_model('primitive_train_tanh.h5')
    training_generator = Mygenerator(batch_size, input_data)

    eval_generator = Mygenerator(batch_size, eval_test)
    #
    model.fit_generator(generator=training_generator,
                        validation_data=eval_generator,
                        workers=6,
                        epochs=5,
                        shuffle=True,
                        # steps_per_epoch=2,
                        use_multiprocessing=True,
                        callbacks=[tensorboard]
                        )
    # model2.fit_generator(generator=training_generator,
    #                     validation_data=eval_generator,
    #                     workers=6,
    #                     epochs=5,
    #                     shuffle=True,
    #                     # steps_per_epoch=2,
    #                     use_multiprocessing=True,
    #                     callbacks=[tensorboard]
    #                     )
    predictions = model.predict_generator(eval_generator)
    score = model.evaluate_generator(eval_generator, workers=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(predictions)

    # model.save('primitive_train_lstm.h5')
    # model.save('primitive_train_lstm2.h5')