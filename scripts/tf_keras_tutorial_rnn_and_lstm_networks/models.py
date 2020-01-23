#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

# --------------------------------- LINKS ------------------------------- #
# - https://www.tensorflow.org/guide/keras/rnn                            #
# - https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model  #
# ----------------------------------------------------------------------- #


from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

import os
import sys

def get_example_lstm_model_sequential(plot_model_flag: bool = False, dir_path: str = '', filename: str = 'example_lstm_model_sequential.png' ):
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units and softmax activation.
    model.add(layers.Dense(10, activation='softmax'))

    if plot_model_flag is True:
        full_path: str = os.path.join(dir_path, filename)
        tf.keras.utils.plot_model(
            model, \
            f'{full_path}', \
            show_shapes=True, \
            expand_nested=True,
            )
    model.summary()
    return model

def get_example_gru_model_sequential(return_sequences_flag:bool = False, plot_model_flag: bool = False, dir_path: str = '', filename: str = 'example_gru_model_sequential.png' ):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=return_sequences_flag))   

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(10, activation='softmax'))
    if plot_model_flag is True:
        full_path: str = os.path.join(dir_path, filename)
        tf.keras.utils.plot_model(
            model, \
            f'{full_path}', \
            show_shapes=True, \
            expand_nested=True,
            )

    model.summary() 

    return model

def get_encoder_decoder_sequence_to_sequence_lstm_model_functional(plot_model_flag: bool = False, dir_path: str = '', filename: str = 'example_ecn_dec_model_functional.png' ):
    encoder_vocab = 1000
    decoder_vocab = 2000

    encoder_input = layers.Input(shape=(None, ))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(
        64, return_state=True, name='encoder')(encoder_embedded)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None, ))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = layers.LSTM(
        64, name='decoder')(decoder_embedded, initial_state=encoder_state)
    output = layers.Dense(10, activation='softmax')(decoder_output)

    model = tf.keras.Model([encoder_input, decoder_input], output)

    if plot_model_flag is True:
        full_path: str = os.path.join(dir_path, filename)
        tf.keras.utils.plot_model(
            model, \
            f'{full_path}', \
            show_shapes=True, \
            expand_nested=True,
        )
    model.summary()

    return model

def get_bidirectional_lstm_model_sequential(plot_model_flag: bool = False, dir_path: str = '', filename: str = 'example_bilstm_model_sequential.png' ):
    model = tf.keras.Sequential()

    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), 
                               input_shape=(5, 10)))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(10, activation='softmax'))
    if plot_model_flag is True:
        full_path: str = os.path.join(dir_path, filename)
        tf.keras.utils.plot_model(
            model, \
            f'{full_path}', \
            show_shapes=True, \
            expand_nested=True,
            )
    model.summary()
    return model


def get_mnist_classifier_made_from_lstm_model(allow_cudnn_kernel: bool = True, plot_model_flag: bool = False, dir_path: str = '', filename: str = 'example_mnist_classifier_lstm_model_sequential.png' ):
    batch_size = 64
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # Each input sequence will be of size (28, 28) (height is treated like time).
    input_dim = 28

    units = 64
    output_size = 10  # labels are from 0 to 9

    model = build_model(units, input_dim, output_size)

    if plot_model_flag is True:
        full_path: str = os.path.join(dir_path, filename)
        tf.keras.utils.plot_model(
            model, \
            f'{full_path}', \
            show_shapes=True, \
            expand_nested=True,
        )
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])

    x_train, y_train, x_test, y_test, _, _= load_mnist_dataset()

    model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)


    return model

# Build the RNN model
def build_model(units, input_dim, output_size, allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not at the cell level.
  # This means `LSTM(units)` will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.
    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))
  model = tf.keras.models.Sequential([
      lstm_layer,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(output_size, activation='softmax')]
  )
  return model

def load_mnist_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]
    return x_train, y_train, x_test, y_test, sample, sample_label